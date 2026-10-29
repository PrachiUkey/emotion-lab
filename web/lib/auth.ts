// Shared-password auth. The cookie holds HMAC(password, tag), never the password itself.
// Uses Web Crypto so it runs in the Edge middleware too.
export const COOKIE = 'el_auth';

export async function sessionToken(): Promise<string> {
  const password = process.env.DASHBOARD_PASSWORD;
  if (!password) throw new Error('DASHBOARD_PASSWORD is not set');
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey('raw', enc.encode(password), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode('emotion-lab-session-v1'));
  return Array.from(new Uint8Array(sig), (b) => b.toString(16).padStart(2, '0')).join('');
}

// Absolute URL on the host the browser actually used. req.url can carry a normalized host (e.g. localhost
// vs 127.0.0.1, or an internal host behind a proxy), and redirecting there would drop the auth cookie.
export function urlOnRequestHost(req: Request, path: string) {
  const h = req.headers;
  const host = h.get('x-forwarded-host') ?? h.get('host') ?? new URL(req.url).host;
  const proto = h.get('x-forwarded-proto') ?? new URL(req.url).protocol.replace(':', '');
  return new URL(path, `${proto}://${host}`);
}

export function safeEqual(a: string, b: string) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
