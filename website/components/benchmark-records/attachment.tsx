import type { RecordSource } from "@/lib/benchmark-records/client";
import type { AssetLink } from "@/lib/benchmark-records/types";
import { useAttachment } from "./use-records";

export function Attachment({
  source,
  link,
}: {
  source: RecordSource;
  link: AssetLink;
}) {
  const asset = useAttachment(source, link);
  if (asset.status === "loading")
    return (
      <p role="status" className="text-xs">
        Loading {link.path}…
      </p>
    );
  if (asset.status === "error")
    return (
      <p role="alert" className="text-xs">
        {asset.message}
      </p>
    );
  return (
    <div className="my-4">
      <a
        className="text-xs text-[var(--lagunita)] underline"
        href={asset.value.url}
        download={link.path}
      >
        Download {link.path} ({link.role})
      </a>
      {asset.value.imageUrl && (
        // Exact released image bytes are supplied as a local Blob URL.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={asset.value.imageUrl}
          alt={link.path}
          className="mt-3 max-h-[28rem] max-w-full rounded object-contain"
        />
      )}
      {link.media_type.startsWith("audio/") && (
        <audio
          controls
          preload="none"
          src={asset.value.url}
          className="mt-3 max-w-full"
        />
      )}
      {link.media_type.startsWith("video/") && (
        <video
          controls
          preload="none"
          src={asset.value.url}
          className="mt-3 max-h-[28rem] max-w-full"
        />
      )}
    </div>
  );
}
