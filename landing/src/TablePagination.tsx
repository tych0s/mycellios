import { ChevronLeft, ChevronRight } from "lucide-react";

export const TABLE_PAGE_SIZE = 10;

export function TablePagination({
  label,
  page,
  pageCount,
  total,
  onPageChange,
}: {
  label: string;
  page: number;
  pageCount: number;
  total: number;
  onPageChange: (page: number) => void;
}) {
  const safePage = Math.min(Math.max(page, 0), Math.max(pageCount - 1, 0));
  const start = total === 0 ? 0 : safePage * TABLE_PAGE_SIZE + 1;
  const end = Math.min(total, (safePage + 1) * TABLE_PAGE_SIZE);
  return <nav className="table-pagination" aria-label={`${label} pagination`}>
    <span>Showing {start}–{end} of {total}</span>
    <div>
      <button type="button" aria-label={`Previous ${label} page`} disabled={safePage === 0} onClick={() => onPageChange(Math.max(0, safePage - 1))}>
        <ChevronLeft size={14} />
      </button>
      <span>Page {safePage + 1} of {Math.max(pageCount, 1)}</span>
      <button type="button" aria-label={`Next ${label} page`} disabled={safePage >= pageCount - 1} onClick={() => onPageChange(Math.min(Math.max(pageCount - 1, 0), safePage + 1))}>
        <ChevronRight size={14} />
      </button>
    </div>
  </nav>;
}
