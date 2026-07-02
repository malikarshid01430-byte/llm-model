type DashboardCardProps = {
  title: string;
  description: string;
};

export default function DashboardCard({ title, description }: DashboardCardProps) {
  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/20">
      <h2 className="text-xl font-semibold text-white">{title}</h2>
      <p className="mt-3 text-slate-400">{description}</p>
    </div>
  );
}
