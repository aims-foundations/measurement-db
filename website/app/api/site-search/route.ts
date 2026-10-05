import { type NextRequest, NextResponse } from "next/server";
import { MAIN_SITE_ORIGIN } from "@/lib/main-site-url";

export async function GET(request: NextRequest) {
  const query = (request.nextUrl.searchParams.get("q") ?? "")
    .slice(0, 120)
    .trim();

  if (query.length < 2) {
    return NextResponse.json({ results: [], total: 0 });
  }

  const searchUrl = new URL("/api/search", MAIN_SITE_ORIGIN);
  searchUrl.searchParams.set("q", query);

  try {
    const response = await fetch(searchUrl, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      throw new Error(`Main-site search returned HTTP ${response.status}`);
    }

    return NextResponse.json(await response.json(), {
      headers: { "Cache-Control": "private, max-age=300" },
    });
  } catch (error) {
    console.error(`[site-search] ${String(error)}`);
    return NextResponse.json(
      { error: "Search is temporarily unavailable" },
      { status: 502 },
    );
  }
}
