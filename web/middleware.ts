import { NextRequest, NextResponse } from 'next/server';
import { COOKIE, safeEqual, sessionToken, urlOnRequestHost } from '@/lib/auth';

export async function middleware(req: NextRequest) {
  const cookie = req.cookies.get(COOKIE)?.value ?? '';
  if (cookie && safeEqual(cookie, await sessionToken())) return NextResponse.next();
  if (req.nextUrl.pathname.startsWith('/api/')) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }
  return NextResponse.redirect(urlOnRequestHost(req, `/login?next=${encodeURIComponent(req.nextUrl.pathname)}`));
}

// Everything except the login page/route and Next static assets requires auth.
export const config = { matcher: ['/((?!login|api/login|_next/static|_next/image|favicon.ico).*)'] };
