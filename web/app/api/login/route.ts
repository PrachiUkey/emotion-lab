import { NextRequest, NextResponse } from 'next/server';
import { COOKIE, safeEqual, sessionToken } from '@/lib/auth';

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const password = String(form.get('password') ?? '');
  const next = String(form.get('next') ?? '/');
  const dest = new URL(next.startsWith('/') && !next.startsWith('//') ? next : '/', req.url);
  if (!safeEqual(password, process.env.DASHBOARD_PASSWORD ?? '\u0000')) {
    return NextResponse.redirect(new URL(`/login?error=1&next=${encodeURIComponent(next)}`, req.url), 303);
  }
  const res = NextResponse.redirect(dest, 303);
  res.cookies.set(COOKIE, await sessionToken(), {
    httpOnly: true, secure: process.env.NODE_ENV === 'production', sameSite: 'lax', path: '/', maxAge: 60 * 60 * 24 * 14,
  });
  return res;
}
