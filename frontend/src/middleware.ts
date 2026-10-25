import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Middleware handling legacy root explorer URLs and /welcome redirects (PLAN_LANDING §8).
 * Redirects /?date=... to /explore?date=... preserving all query params,
 * and /welcome to / root landing page.
 */
export function middleware(request: NextRequest) {
  const { pathname, searchParams } = request.nextUrl;

  // 1. Redirect /welcome to root /
  if (pathname === "/welcome") {
    const url = request.nextUrl.clone();
    url.pathname = "/";
    return NextResponse.redirect(url, 308);
  }

  // 2. Redirect legacy shared links on root (/?date=... -> /explore?date=...)
  if (pathname === "/") {
    const hasExplorerParams =
      searchParams.has("date") ||
      searchParams.has("depth") ||
      searchParams.has("depth_m") ||
      searchParams.has("depth_index") ||
      searchParams.has("field") ||
      searchParams.has("lat") ||
      searchParams.has("lon") ||
      searchParams.has("view") ||
      searchParams.has("layer");

    if (hasExplorerParams) {
      const url = request.nextUrl.clone();
      url.pathname = "/explore";
      return NextResponse.redirect(url, 307);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/", "/welcome"],
};
