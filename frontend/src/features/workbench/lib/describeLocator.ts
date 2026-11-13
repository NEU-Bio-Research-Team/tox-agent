/** A passage locator (`table-wrap:1/row:3/cell:2`) said the way a reader would:
 * "Bảng 1, hàng 3, ô 2". An unknown shape is "đoạn trích", never a raw path. */
export function describeLocator(locator: string): string {
  const table = /table[^/:]*:(\d+)\/row:(\d+)\/cell:(\d+)/.exec(locator);
  if (table) return `Bảng ${table[1]}, hàng ${table[2]}, ô ${table[3]}`;
  const row = /table[^/:]*:(\d+)\/row:(\d+)/.exec(locator);
  if (row) return `Bảng ${row[1]}, hàng ${row[2]}`;
  return 'đoạn trích';
}
