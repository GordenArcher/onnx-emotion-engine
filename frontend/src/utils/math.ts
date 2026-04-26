export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((acc, val) => acc + val, 0);
  return exps.map((e) => e / sumExps);
}
