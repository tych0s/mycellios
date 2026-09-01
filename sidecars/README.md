# Native sidecars

This directory contains narrowly scoped helpers built and shipped as part of
Mycellios. Sidecars must not provide a generic escape hatch to an external
inference runtime.

`windows-job-broker` supplies operating-system process isolation for the native
node on Windows. Its package and lifecycle are verified by the normal native
product gates.
