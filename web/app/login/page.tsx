export default function Login({ searchParams }: { searchParams: { next?: string; error?: string } }) {
  return (
    <main className="login">
      <form method="post" action="/api/login" className="card">
        <h1>Emotion Lab</h1>
        <p className="muted">Team access only.</p>
        <input type="hidden" name="next" value={searchParams.next ?? '/'} />
        <input type="password" name="password" placeholder="Password" autoFocus required />
        {searchParams.error && <p className="error">Wrong password.</p>}
        <button type="submit">Enter</button>
      </form>
    </main>
  );
}
