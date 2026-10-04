using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

internal static class Program
{
    private const string BrokerSchema = "mycellios-windows-job-broker/1";
    private const string BrokerProbe = "mycellios-windows-job-broker/1:ready";
    private const uint CreateSuspended = 0x00000004;
    private const uint CreateNoWindow = 0x08000000;
    private const uint StartfUseStdHandles = 0x00000100;
    private const uint JobObjectLimitKillOnJobClose = 0x00002000;
    private const int JobObjectExtendedLimitInformationClass = 9;
    private const uint Synchronize = 0x00100000;
    private const uint Infinite = 0xFFFFFFFF;
    private const uint WaitObject0 = 0x00000000;
    private const uint WaitFailed = 0xFFFFFFFF;
    private const uint WaitTimeout = 0x00000102;
    private const uint HandleFlagInherit = 0x00000001;
    private const int StdInputHandle = -10;
    private const int StdOutputHandle = -11;
    private const int StdErrorHandle = -12;
    private const uint GenericRead = 0x80000000;
    private const uint GenericWrite = 0x40000000;
    private const uint FileShareRead = 0x00000001;
    private const uint FileShareWrite = 0x00000002;
    private const uint OpenExisting = 3;
    private const uint FileAttributeNormal = 0x00000080;
    private const uint ParentExitedCode = 0xE0000001;
    private const uint Th32csSnapProcess = 0x00000002;
    private static readonly IntPtr InvalidHandleValue = new(-1);

    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 1 && args[0] == "--probe")
            {
                Console.Out.WriteLine(BrokerProbe);
                return 0;
            }

            if (args.Length != 1)
            {
                throw new InvalidOperationException("exactly_one_request_path_is_required");
            }

            var request = ReadAndDeleteRequest(args[0]);
            return Run(request);
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(
                $"mycellios_windows_job_broker_failed:{NormalizeError(error.Message)}"
            );
            return 70;
        }
    }

    private static BrokerRequest ReadAndDeleteRequest(string requestPathValue)
    {
        if (
            string.IsNullOrWhiteSpace(requestPathValue)
            || requestPathValue != requestPathValue.Trim()
            || requestPathValue.IndexOfAny(['\0', '\r', '\n']) >= 0
            || !Path.IsPathFullyQualified(requestPathValue)
        )
        {
            throw new InvalidOperationException("request_path_is_invalid");
        }

        var requestPath = Path.GetFullPath(requestPathValue);
        var attributes = File.GetAttributes(requestPath);
        if ((attributes & FileAttributes.ReparsePoint) != 0)
        {
            throw new InvalidOperationException("request_path_must_not_be_a_reparse_point");
        }

        string json;
        try
        {
            json = File.ReadAllText(requestPath, Encoding.UTF8);
        }
        finally
        {
            File.Delete(requestPath);
        }

        using var document = JsonDocument.Parse(
            json,
            new JsonDocumentOptions
            {
                AllowTrailingCommas = false,
                CommentHandling = JsonCommentHandling.Disallow,
                MaxDepth = 8,
            }
        );
        return ParseRequest(document.RootElement);
    }

    private static BrokerRequest ParseRequest(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidOperationException("request_must_be_an_object");
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in root.EnumerateObject())
        {
            if (
                !seen.Add(property.Name)
                || property.Name is not ("schema" or "executable" or "args" or "cwd" or "parentPid")
            )
            {
                throw new InvalidOperationException("request_has_unknown_or_duplicate_fields");
            }
        }
        if (seen.Count != 5)
        {
            throw new InvalidOperationException("request_has_missing_fields");
        }

        var schema = RequiredString(root, "schema", 128);
        if (schema != BrokerSchema)
        {
            throw new InvalidOperationException("request_schema_is_invalid");
        }

        var executable = RequiredAbsolutePath(root, "executable");
        if (!File.Exists(executable))
        {
            throw new InvalidOperationException("target_executable_does_not_exist");
        }

        var cwd = RequiredAbsolutePath(root, "cwd");
        if (!Directory.Exists(cwd))
        {
            throw new InvalidOperationException("target_working_directory_does_not_exist");
        }

        var argsElement = root.GetProperty("args");
        if (argsElement.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidOperationException("target_args_must_be_an_array");
        }
        var arguments = new List<string>();
        foreach (var item in argsElement.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String || arguments.Count >= 512)
            {
                throw new InvalidOperationException("target_args_are_invalid");
            }
            var argument = item.GetString()
                ?? throw new InvalidOperationException("target_arg_is_invalid");
            if (argument.Length > 32_000 || argument.IndexOf('\0') >= 0)
            {
                throw new InvalidOperationException("target_arg_is_invalid");
            }
            arguments.Add(argument);
        }

        var parentElement = root.GetProperty("parentPid");
        if (
            parentElement.ValueKind != JsonValueKind.Number
            || !parentElement.TryGetInt32(out var parentPid)
            || parentPid < 1
            || parentPid == Environment.ProcessId
        )
        {
            throw new InvalidOperationException("trusted_parent_pid_is_invalid");
        }

        return new BrokerRequest(executable, arguments, cwd, parentPid);
    }

    private static string RequiredAbsolutePath(JsonElement root, string name)
    {
        var value = RequiredString(root, name, 32_000);
        if (
            value != value.Trim()
            || value.IndexOfAny(['\0', '\r', '\n']) >= 0
            || !Path.IsPathFullyQualified(value)
        )
        {
            throw new InvalidOperationException($"{name}_is_invalid");
        }
        return Path.GetFullPath(value);
    }

    private static string RequiredString(JsonElement root, string name, int maximumLength)
    {
        var element = root.GetProperty(name);
        if (element.ValueKind != JsonValueKind.String)
        {
            throw new InvalidOperationException($"{name}_must_be_a_string");
        }
        var value = element.GetString()
            ?? throw new InvalidOperationException($"{name}_must_be_a_string");
        if (value.Length == 0 || value.Length > maximumLength)
        {
            throw new InvalidOperationException($"{name}_is_invalid");
        }
        return value;
    }

    private static int Run(BrokerRequest request)
    {
        IntPtr job = IntPtr.Zero;
        IntPtr parent = IntPtr.Zero;
        IntPtr childProcess = IntPtr.Zero;
        IntPtr childThread = IntPtr.Zero;
        var openedStandardHandles = new List<IntPtr>();
        var resumed = false;

        try
        {
            if (request.ParentPid != TrustedParentProcessId())
            {
                throw new InvalidOperationException(
                    "trusted_parent_pid_does_not_match_broker_parent"
                );
            }
            parent = OpenProcess(Synchronize, false, request.ParentPid);
            EnsureHandle(parent, "open_trusted_parent");

            job = CreateJobObjectW(IntPtr.Zero, null);
            EnsureHandle(job, "create_job_object");
            ConfigureKillOnClose(job);

            var startup = new StartupInfo
            {
                cb = Marshal.SizeOf<StartupInfo>(),
                dwFlags = StartfUseStdHandles,
                hStdInput = InheritableStandardHandle(StdInputHandle, GenericRead, openedStandardHandles),
                hStdOutput = InheritableStandardHandle(StdOutputHandle, GenericWrite, openedStandardHandles),
                hStdError = InheritableStandardHandle(StdErrorHandle, GenericWrite, openedStandardHandles),
            };
            var commandLine = BuildCommandLine(request.Executable, request.Arguments);
            if (commandLine.Length > 32_767)
            {
                throw new InvalidOperationException("target_command_line_is_too_long");
            }

            if (
                !CreateProcessW(
                    request.Executable,
                    new StringBuilder(commandLine),
                    IntPtr.Zero,
                    IntPtr.Zero,
                    true,
                    CreateSuspended | CreateNoWindow,
                    IntPtr.Zero,
                    request.WorkingDirectory,
                    ref startup,
                    out var processInformation
                )
            )
            {
                ThrowWin32("create_suspended_target");
            }
            childProcess = processInformation.hProcess;
            childThread = processInformation.hThread;

            if (!AssignProcessToJobObject(job, childProcess))
            {
                ThrowWin32("assign_target_to_job");
            }
            var parentStateBeforeResume = WaitForSingleObject(parent, 0);
            if (parentStateBeforeResume == WaitFailed)
            {
                ThrowWin32("check_trusted_parent_before_resume");
            }
            if (parentStateBeforeResume != WaitTimeout)
            {
                throw new InvalidOperationException(
                    "trusted_parent_exited_before_target_resume"
                );
            }
            if (ResumeThread(childThread) == uint.MaxValue)
            {
                ThrowWin32("resume_job_target");
            }
            resumed = true;

            var waitHandles = new[] { childProcess, parent };
            var wait = WaitForMultipleObjects(
                (uint)waitHandles.Length,
                waitHandles,
                false,
                Infinite
            );
            if (wait == WaitFailed)
            {
                ThrowWin32("wait_for_target_or_parent");
            }
            if (wait == WaitObject0 + 1)
            {
                if (!TerminateJobObject(job, ParentExitedCode))
                {
                    ThrowWin32("terminate_job_after_parent_exit");
                }
                _ = WaitForSingleObject(childProcess, Infinite);
                return unchecked((int)ParentExitedCode);
            }
            if (wait != WaitObject0)
            {
                throw new InvalidOperationException($"unexpected_wait_result_{wait}");
            }
            if (!GetExitCodeProcess(childProcess, out var exitCode))
            {
                ThrowWin32("read_target_exit_code");
            }
            return unchecked((int)exitCode);
        }
        finally
        {
            if (!resumed && job != IntPtr.Zero && job != InvalidHandleValue)
            {
                _ = TerminateJobObject(job, ParentExitedCode);
            }
            CloseIfValid(childThread);
            CloseIfValid(childProcess);
            CloseIfValid(parent);
            CloseIfValid(job);
            foreach (var handle in openedStandardHandles)
            {
                CloseIfValid(handle);
            }
        }
    }

    private static int TrustedParentProcessId()
    {
        var snapshot = CreateToolhelp32Snapshot(Th32csSnapProcess, 0);
        EnsureHandle(snapshot, "snapshot_process_table");
        try
        {
            var entry = new ProcessEntry32
            {
                dwSize = (uint)Marshal.SizeOf<ProcessEntry32>(),
            };
            if (!Process32FirstW(snapshot, ref entry))
            {
                ThrowWin32("read_first_process_entry");
            }
            do
            {
                if (entry.th32ProcessID == (uint)Environment.ProcessId)
                {
                    if (entry.th32ParentProcessID is 0 or > int.MaxValue)
                    {
                        throw new InvalidOperationException(
                            "broker_parent_pid_is_invalid"
                        );
                    }
                    return (int)entry.th32ParentProcessID;
                }
            } while (Process32NextW(snapshot, ref entry));
            throw new InvalidOperationException(
                "broker_process_is_missing_from_process_table"
            );
        }
        finally
        {
            CloseIfValid(snapshot);
        }
    }

    private static void ConfigureKillOnClose(IntPtr job)
    {
        var limits = new JobObjectExtendedLimitInformation
        {
            BasicLimitInformation = new JobObjectBasicLimitInformation
            {
                LimitFlags = JobObjectLimitKillOnJobClose,
            },
        };
        if (
            !SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformationClass,
                ref limits,
                (uint)Marshal.SizeOf<JobObjectExtendedLimitInformation>()
            )
        )
        {
            ThrowWin32("configure_job_kill_on_close");
        }
    }

    private static IntPtr InheritableStandardHandle(
        int standardHandle,
        uint nulAccess,
        List<IntPtr> openedHandles
    )
    {
        var handle = GetStdHandle(standardHandle);
        if (handle == IntPtr.Zero || handle == InvalidHandleValue)
        {
            var attributes = new SecurityAttributes
            {
                nLength = Marshal.SizeOf<SecurityAttributes>(),
                bInheritHandle = true,
            };
            handle = CreateFileW(
                "NUL",
                nulAccess,
                FileShareRead | FileShareWrite,
                ref attributes,
                OpenExisting,
                FileAttributeNormal,
                IntPtr.Zero
            );
            EnsureHandle(handle, "open_nul_standard_handle");
            openedHandles.Add(handle);
        }
        if (!SetHandleInformation(handle, HandleFlagInherit, HandleFlagInherit))
        {
            ThrowWin32("make_standard_handle_inheritable");
        }
        return handle;
    }

    private static string BuildCommandLine(string executable, IReadOnlyList<string> arguments)
    {
        var values = new List<string>(arguments.Count + 1) { executable };
        values.AddRange(arguments);
        return string.Join(" ", values.Select(QuoteWindowsArgument));
    }

    private static string QuoteWindowsArgument(string value)
    {
        if (value.Length > 0 && value.All(character =>
            character is not (' ' or '\t' or '"' or '\v' or '\n')))
        {
            return value;
        }

        var result = new StringBuilder(value.Length + 2);
        result.Append('"');
        var backslashes = 0;
        foreach (var character in value)
        {
            if (character == '\\')
            {
                backslashes += 1;
                continue;
            }
            if (character == '"')
            {
                result.Append('\\', backslashes * 2 + 1);
                result.Append('"');
                backslashes = 0;
                continue;
            }
            result.Append('\\', backslashes);
            backslashes = 0;
            result.Append(character);
        }
        result.Append('\\', backslashes * 2);
        result.Append('"');
        return result.ToString();
    }

    private static void EnsureHandle(IntPtr handle, string operation)
    {
        if (handle == IntPtr.Zero || handle == InvalidHandleValue)
        {
            ThrowWin32(operation);
        }
    }

    private static void ThrowWin32(string operation)
    {
        throw new Win32Exception(
            Marshal.GetLastWin32Error(),
            $"{operation}_failed"
        );
    }

    private static void CloseIfValid(IntPtr handle)
    {
        if (handle != IntPtr.Zero && handle != InvalidHandleValue)
        {
            _ = CloseHandle(handle);
        }
    }

    private static string NormalizeError(string value)
    {
        var normalized = value.Replace('\r', ' ').Replace('\n', ' ').Replace('\0', ' ');
        return normalized.Length <= 1_000 ? normalized : normalized[..1_000];
    }

    private sealed record BrokerRequest(
        string Executable,
        IReadOnlyList<string> Arguments,
        string WorkingDirectory,
        int ParentPid
    );

    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes
    {
        public int nLength;
        public IntPtr lpSecurityDescriptor;
        [MarshalAs(UnmanagedType.Bool)]
        public bool bInheritHandle;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct StartupInfo
    {
        public int cb;
        public string? lpReserved;
        public string? lpDesktop;
        public string? lpTitle;
        public uint dwX;
        public uint dwY;
        public uint dwXSize;
        public uint dwYSize;
        public uint dwXCountChars;
        public uint dwYCountChars;
        public uint dwFillAttribute;
        public uint dwFlags;
        public ushort wShowWindow;
        public ushort cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation
    {
        public IntPtr hProcess;
        public IntPtr hThread;
        public uint dwProcessId;
        public uint dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private unsafe struct ProcessEntry32
    {
        public uint dwSize;
        public uint cntUsage;
        public uint th32ProcessID;
        public nuint th32DefaultHeapID;
        public uint th32ModuleID;
        public uint cntThreads;
        public uint th32ParentProcessID;
        public int pcPriClassBase;
        public uint dwFlags;
        public fixed char szExeFile[260];
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public nuint MinimumWorkingSetSize;
        public nuint MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public nuint Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInformation
    {
        public JobObjectBasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public nuint ProcessMemoryLimit;
        public nuint JobMemoryLimit;
        public nuint PeakProcessMemoryUsed;
        public nuint PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObjectW(IntPtr jobAttributes, string? name);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        int informationClass,
        ref JobObjectExtendedLimitInformation information,
        uint informationLength
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateProcessW(
        string applicationName,
        StringBuilder commandLine,
        IntPtr processAttributes,
        IntPtr threadAttributes,
        [MarshalAs(UnmanagedType.Bool)] bool inheritHandles,
        uint creationFlags,
        IntPtr environment,
        string currentDirectory,
        ref StartupInfo startupInfo,
        out ProcessInformation processInformation
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(
        uint desiredAccess,
        [MarshalAs(UnmanagedType.Bool)] bool inheritHandle,
        int processId
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForMultipleObjects(
        uint count,
        [In] IntPtr[] handles,
        [MarshalAs(UnmanagedType.Bool)] bool waitAll,
        uint milliseconds
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetStdHandle(int standardHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateFileW(
        string fileName,
        uint desiredAccess,
        uint shareMode,
        ref SecurityAttributes securityAttributes,
        uint creationDisposition,
        uint flagsAndAttributes,
        IntPtr templateFile
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CloseHandle(IntPtr handle);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr CreateToolhelp32Snapshot(uint flags, uint processId);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool Process32FirstW(
        IntPtr snapshot,
        ref ProcessEntry32 entry
    );

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool Process32NextW(
        IntPtr snapshot,
        ref ProcessEntry32 entry
    );
}
