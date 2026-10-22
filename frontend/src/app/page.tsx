export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          OceanScope AI
        </h1>
        <p className="text-lg text-slate-400">
          Reconstructing daily 3D subsurface ocean temperature fields across the North Indian Ocean.
        </p>
      </div>
    </main>
  );
}
