import { BookOpen, GraduationCap, Recycle, Users } from "lucide-react";

const teamMembers = [
  { name: "ATISHAY JAIN", rollNumber: "31", enrollmentNumber: "0176AL251033" },
  { name: "ANISH KUMAR", rollNumber: "21", enrollmentNumber: "0176AL251022" },
  { name: "ABHIJEET KUMAR ANAND", rollNumber: "07", enrollmentNumber: "0176AL251007" },
];

export function SiteFooter() {
  return (
    <footer className="relative overflow-hidden border-t border-emerald-900/70 bg-slate-950 text-slate-300">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,rgba(16,185,129,0.17),transparent_48%)]" />

      <div className="relative mx-auto max-w-7xl px-4 pb-6 pt-12 sm:px-6 lg:pt-14">
        <div className="grid gap-10 border-b border-white/10 pb-10 sm:grid-cols-2 lg:grid-cols-[1.25fr_1fr_1fr_1fr] lg:gap-8">
          <div>
            <div className="flex items-center gap-3">
              <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-500 text-white shadow-lg shadow-emerald-950/40">
                <Recycle className="h-6 w-6" aria-hidden="true" />
              </span>
              <div>
                <p className="text-xl font-extrabold tracking-tight text-white">
                  Smart<span className="text-emerald-400">Recycle</span>
                </p>
                <p className="text-xs font-medium uppercase tracking-[0.18em] text-emerald-200/70">
                  Waste Classification
                </p>
              </div>
            </div>
            <p className="mt-4 max-w-sm text-sm leading-6 text-slate-400">
              An AI-based waste classification mini project that encourages
              informed sorting and responsible recycling.
            </p>
            <span className="mt-4 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs font-semibold text-emerald-200">
              <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
              B.Tech CSE - AIML Mini Project
            </span>
          </div>

          <div>
            <h2 className="flex items-center gap-2 text-sm font-bold text-white">
              <GraduationCap className="h-4 w-4 text-emerald-400" aria-hidden="true" />
              Academic Project
            </h2>
            <dl className="mt-4 space-y-3 text-sm">
              <div>
                <dt className="text-xs text-slate-500">Institution</dt>
                <dd className="mt-0.5 font-medium text-slate-200">LNCT Group of Colleges</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">College</dt>
                <dd className="mt-0.5 font-medium text-slate-200">LNCTE College</dd>
              </div>
              <div>
                <dt className="text-xs text-slate-500">Class / Section</dt>
                <dd className="mt-0.5 font-medium text-slate-200">3A</dd>
              </div>
            </dl>
          </div>

          <div>
            <h2 className="flex items-center gap-2 text-sm font-bold text-white">
              <Users className="h-4 w-4 text-emerald-400" aria-hidden="true" />
              Project Team
            </h2>
            <p className="mt-4 text-xs text-slate-500">Team Dark World</p>
            <ul className="mt-3 space-y-3">
              {teamMembers.map((member) => (
                <li key={member.enrollmentNumber} className="text-sm">
                  <p className="font-semibold text-slate-200">{member.name}</p>
                  <p className="mt-0.5 text-xs leading-5 text-slate-400">
                    Roll No. {member.rollNumber} <span className="text-slate-600">·</span> Enrollment No. {member.enrollmentNumber}
                  </p>
                </li>
              ))}
            </ul>
            <p className="mt-4 text-xs text-slate-500">Project Guide</p>
            <p className="mt-1 text-sm font-medium text-slate-200">Prof. Neelesh Gour</p>
          </div>

          <div>
            <h2 className="text-sm font-bold text-white">Developed By</h2>
            <p className="mt-4 text-base font-semibold text-slate-100">Anish Kumar</p>
            <p className="mt-1 text-sm leading-6 text-slate-400">
              Founder of <span className="font-semibold text-emerald-300">AnishNova Technologies</span>
            </p>
            <p className="mt-4 text-xs leading-5 text-slate-500">
              Built as an academic mini project for learning, demonstration,
              and environmental awareness.
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-2 pt-5 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <p>SmartRecycle · LNCT Group of Colleges</p>
          <p>Created by Team Dark World</p>
        </div>
      </div>
    </footer>
  );
}
