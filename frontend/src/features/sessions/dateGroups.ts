export interface DateGroup<T> {
  label: string;
  rows: T[];
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** Sidebar grouping per plan section 8.1: "Hôm nay / Hôm qua / Trước đó" by
 * `updated_at`. Rows are assumed already sorted newest-first by the caller. */
export function groupByRecency<T>(rows: T[], getIso: (row: T) => string): Array<DateGroup<T>> {
  const today = startOfDay(new Date());
  const yesterday = today - 86_400_000;

  const buckets: Record<'today' | 'yesterday' | 'earlier', T[]> = {
    today: [],
    yesterday: [],
    earlier: [],
  };

  for (const row of rows) {
    const day = startOfDay(new Date(getIso(row)));
    if (day === today) buckets.today.push(row);
    else if (day === yesterday) buckets.yesterday.push(row);
    else buckets.earlier.push(row);
  }

  const groups: Array<DateGroup<T>> = [];
  if (buckets.today.length > 0) groups.push({ label: 'Hôm nay', rows: buckets.today });
  if (buckets.yesterday.length > 0) groups.push({ label: 'Hôm qua', rows: buckets.yesterday });
  if (buckets.earlier.length > 0) groups.push({ label: 'Trước đó', rows: buckets.earlier });
  return groups;
}

export function relativeTimeVi(iso: string): string {
  const then = new Date(iso).getTime();
  const diffMs = Date.now() - then;
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return 'vừa xong';
  if (minutes < 60) return `${minutes} phút trước`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} giờ trước`;
  const days = Math.round(hours / 24);
  return `${days} ngày trước`;
}
