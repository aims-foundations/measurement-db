import type { AsyncBuffer, Compressors } from "hyparquet";

/** A small row can share an enormous dictionary/page with other source records. */
export function budgetSource(
  file: AsyncBuffer,
  codecs: Compressors,
  limit = 16 * 1024 * 1024,
) {
  let bytes = 0;
  const reserve = (length: number) => {
    bytes += length;
    if (bytes > limit)
      throw new Error(
        "This record exceeds the 16 MiB source-data preview limit. Download the source table to inspect it.",
      );
  };
  return {
    file: {
      byteLength: file.byteLength,
      slice(start: number, end = file.byteLength) {
        reserve(end - start);
        return file.slice(start, end);
      },
    },
    compressors: Object.fromEntries(
      Object.entries(codecs).map(([codec, decompress]) => [
        codec,
        (input: Uint8Array, length: number) => {
          reserve(length);
          return decompress(input, length);
        },
      ]),
    ) as Compressors,
  };
}
