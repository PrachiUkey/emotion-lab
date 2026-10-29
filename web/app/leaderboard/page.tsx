import { getJson } from '@/lib/store';
import type { Leaderboard } from '@/lib/types';

export const dynamic = 'force-dynamic';

const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`);

export default async function LeaderboardPage() {
  const lb = await getJson<Leaderboard>('eval/leaderboard.json');
  if (!lb) return <main><p className="error">No eval/leaderboard.json yet. Run scripts/evaluate.py.</p></main>;
  const present = Object.keys(lb.gold_distribution);
  return (
    <main>
      <h2 style={{ marginTop: 0 }}>Model leaderboard</h2>
      <p className="muted">
        {lb.n_clips} human-labelled clips (min label confidence: {lb.min_confidence}) ·{' '}
        {present.map((k) => `${k} ${lb.gold_distribution[k]}`).join(' · ')}
      </p>
      <div className="note">
        Ranked by macro-F1 (each emotion counts equally — the data is mostly neutral, so accuracy alone flatters
        models that guess neutral). <b>Neutral FP</b> = share of human-neutral clips the model called emotional;
        talking faces trigger this. With this few labels the confidence intervals are wide — treat close scores as ties.
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr>
            <th>Model</th><th>Macro-F1</th><th>95% CI</th><th>Accuracy</th><th>Neutral FP</th><th>ECE</th><th>Frame flip rate</th>
            {present.map((k) => <th key={k}>recall {k}</th>)}
          </tr></thead>
          <tbody>
            {lb.leaderboard.map((r) => (
              <tr key={r.model}>
                <td><b>{r.model}</b></td>
                <td>{r.macro_f1.toFixed(3)}</td>
                <td className="muted">{r.macro_f1_ci95[0].toFixed(2)}–{r.macro_f1_ci95[1].toFixed(2)}</td>
                <td>{pct(r.accuracy)}</td>
                <td>{pct(r.neutral_fp_rate)}</td>
                <td>{r.ece.toFixed(3)}</td>
                <td>{pct(r.flip_rate)}</td>
                {present.map((k) => <td key={k}>{pct(r.recall[k])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Agreement between models</h3>
      <p className="muted">Cohen&apos;s κ per pair; Fleiss&apos; κ across all: {lb.agreement.fleiss_kappa?.toFixed(3) ?? '—'}</p>
      <div className="table-wrap" style={{ maxWidth: 520 }}>
        <table><tbody>
          {Object.entries(lb.agreement.cohen_kappa).map(([pair, k]) => (
            <tr key={pair}><td>{pair.replace('|', ' vs ')}</td><td>{k.toFixed(3)}</td></tr>
          ))}
        </tbody></table>
      </div>

      <h3>Confusion matrices (rows = human, columns = model)</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: 16 }}>
        {lb.leaderboard.map((r) => {
          const cols = lb.labels.map((_, j) => j).filter((j) => r.confusion.some((row) => row[j] > 0) || present.includes(lb.labels[j]));
          return (
            <div key={r.model} className="card">
              <b>{r.model}</b>
              <div className="table-wrap" style={{ marginTop: 8 }}>
                <table className="cm">
                  <thead><tr><th /> {cols.map((j) => <th key={j}>{lb.labels[j].slice(0, 5)}</th>)}</tr></thead>
                  <tbody>
                    {lb.labels.map((l, i) => present.includes(l) && (
                      <tr key={l}><th>{l}</th>{cols.map((j) => (
                        <td key={j} style={{ background: r.confusion[i][j] ? (i === j ? 'rgba(76,195,138,.35)' : 'rgba(239,107,94,.25)') : undefined }}>
                          {r.confusion[i][j] || ''}
                        </td>
                      ))}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </div>
    </main>
  );
}
