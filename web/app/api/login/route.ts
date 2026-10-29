import { NextRequest, NextResponse } from 'next/server';
import { COOKIE, safeEqual, sessionToken, urlOnRequestHost } from '@/lib/auth';

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const password = String(form.get('password') ?? '');
  const raw = String(form.get('next') ?? '/');
  const next = raw.startsWith('/') && !raw.startsWith('//') ? raw : '/';
  if (!safeEqual(password, process.env.DASHBOARD_PASSWORD ?? '\u0000')) {
    return NextResponse.redirect(urlOnRequestHost(req, `/login?error=1&next=${encodeURIComponent(next)}`), 303);
  }
  const res = NextResponse.redirect(urlOnRequestHost(req, next), 303);
  res.cookies.set(COOKIE, await sessionToken(), {
    httpOnly: true, secure: process.env.NODE_ENV === 'production', sameSite: 'lax', path: '/', maxAge: 60 * 60 * 24 * 14,
  });
  return res;
}
