// Decorative scene coordinates, never scientific positions or observations.
export const fishLayout = Array.from({ length: 18 }, (_, i) => {
  const x = 0.61 + ((i * 0.61803398875) % 1) * 0.39;
  return {
    x,
    y: 0.79 - (x - 0.55) * 0.49 + Math.sin(i * 2.1) * 0.04,
    width: 0.029 + ((i * 7) % 5) * 0.006,
    opacity: 0.3 + (i % 4) * 0.13,
  };
});
