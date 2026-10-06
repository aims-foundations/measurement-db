import { permanentRedirect } from "next/navigation";

export const dynamic = "force-static";

export default function SchemaDefinitionPage() {
  permanentRedirect("/#schema");
}
