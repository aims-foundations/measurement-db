export const TEXT_PART_SIZE = 32768;

/** Slice at UTF-16 boundaries without splitting a surrogate pair or dropping text. */
export function textPart(text: string, page: number): string {
  const boundary = (index: number) => {
    const code = text.charCodeAt(index);
    return code >= 0xdc00 && code <= 0xdfff ? index - 1 : index;
  };
  return text.slice(
    boundary(page * TEXT_PART_SIZE),
    boundary(Math.min(text.length, (page + 1) * TEXT_PART_SIZE)),
  );
}
