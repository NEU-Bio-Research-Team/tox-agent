import { useState, type ReactNode } from 'react';
import { Link } from 'react-router';
import { BookOpenText, FileText } from 'lucide-react';
import { Navbar } from '../../app/shell/Navbar';
import { Footer } from '../../app/shell/Footer';

const SECTION_ITEMS = [
  { id: 'intro', label: '1. Introduction' },
  { id: 'quick-analysis', label: '2. Quick Analysis' },
  { id: 'results', label: '3. Reading the Results' },
  { id: 'sessions', label: '4. Sessions and the Agent' },
  { id: 'ai-connections', label: '5. AI Connections' },
  { id: 'limitations', label: '6. Notes & Limitations' },
] as const;

const CARD_STYLE = { backgroundColor: 'var(--surface)', border: '1px solid var(--border)' } as const;
const BODY_STYLE = { color: 'var(--text-muted)', lineHeight: '1.8' } as const;

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 space-y-4">
      <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
        {title}
      </h2>
      <div className="space-y-3 rounded-xl p-6" style={CARD_STYLE}>
        {children}
      </div>
    </section>
  );
}

/**
 * The user guide, in the production site's documents layout. The text
 * describes this version of the product: the previous guide walked through a
 * report with a single risk level, which no longer exists.
 */
export function DocumentsPage() {
  const [activeSection, setActiveSection] = useState<string>('intro');

  const handleScrollTo = (sectionId: string) => {
    setActiveSection(sectionId);
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="legacy-ui" style={{ minHeight: '100vh', backgroundColor: 'var(--bg)' }}>
      <Navbar />

      <div className="mx-auto grid max-w-[1480px] grid-cols-1 lg:grid-cols-[280px_1fr]">
        <aside
          className="border-b p-6 lg:sticky lg:top-16 lg:h-[calc(100vh-4rem)] lg:overflow-y-auto lg:border-b-0 lg:border-r"
          style={{ borderColor: 'var(--border)' }}
        >
          <div className="mb-8">
            <div className="mb-3 flex items-center gap-2">
              <BookOpenText className="h-4 w-4" style={{ color: 'var(--accent-blue)' }} />
              <h2 className="text-sm font-semibold uppercase" style={{ color: 'var(--text)', letterSpacing: '0.05em' }}>
                Documents
              </h2>
            </div>
            <div
              className="rounded-lg px-3 py-2"
              style={{ backgroundColor: 'var(--accent-blue-muted)', color: 'var(--accent-blue)' }}
            >
              <div className="flex items-start gap-2">
                <FileText className="mt-0.5 h-4 w-4" />
                <div>
                  <p className="text-sm font-semibold">User Guide</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    How to screen a molecule and read what comes back.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="border-t pt-6" style={{ borderColor: 'var(--border)' }}>
            <h3 className="mb-3 text-xs font-semibold uppercase" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
              On This Page
            </h3>
            <nav className="space-y-2" aria-label="On this page">
              {SECTION_ITEMS.map((section) => (
                <button
                  key={section.id}
                  type="button"
                  onClick={() => handleScrollTo(section.id)}
                  aria-current={activeSection === section.id ? 'true' : undefined}
                  className="w-full rounded-lg px-3 py-2 text-left text-sm transition-colors"
                  style={{
                    backgroundColor: activeSection === section.id ? 'var(--accent-blue-muted)' : 'transparent',
                    color: activeSection === section.id ? 'var(--accent-blue)' : 'var(--text-muted)',
                  }}
                >
                  {activeSection === section.id ? '●' : '○'} {section.label}
                </button>
              ))}
            </nav>
          </div>
        </aside>

        <main className="max-w-[960px] space-y-10 p-6 md:p-10">
          <header className="rounded-xl p-6" style={CARD_STYLE}>
            <p className="mb-2 text-xs uppercase" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
              ToxAgent Documentation
            </p>
            <h1 className="mb-2 text-3xl font-bold" style={{ color: 'var(--text)' }}>
              User Guide
            </h1>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Screening a molecule from its SMILES string, and following the result into an investigation.
            </p>
          </header>

          <Section id="intro" title="1. Introduction to ToxAgent">
            <p style={BODY_STYLE}>
              ToxAgent screens a molecule against three independent toxicity measurements: hERG channel blockade,
              the twelve Tox21 assays, and the ClinTox clinical-trial signal. Each one is reported on its own, with
              the model that produced it and the threshold that was applied.
            </p>
            <p style={BODY_STYLE}>
              There are two ways to work. Quick analysis answers one question about one molecule and keeps nothing.
              A session keeps the analyses, lets you question an agent about them, and records where every number in
              an answer came from.
            </p>
          </Section>

          <Section id="quick-analysis" title="2. Quick Analysis">
            <p style={BODY_STYLE}>
              Open <Link to="/analyze" className="underline" style={{ color: 'var(--accent-blue)' }}>the analysis page</Link>{' '}
              and give it a structure in one of three ways:
            </p>
            <ul className="list-disc space-y-2 pl-5" style={{ color: 'var(--text-muted)' }}>
              <li><strong>Type SMILES.</strong> Paste or type the string, or pick one of the examples. Switch to Batch to send a list, one SMILES per line; a line that fails is reported by its position and the rest still run.</li>
              <li><strong>Draw Molecule.</strong> Draw the structure, then choose Use Drawn SMILES.</li>
              <li><strong>Upload Image.</strong> Upload a picture of a structure and extract its SMILES. Recognition is a guess: check the string and correct it before analysing. The tab says so when a deployment has recognition switched off.</li>
            </ul>
            <p style={BODY_STYLE}>
              Advanced options lists the endpoints and models this deployment serves. You can choose which endpoints
              to run, pick a model for each, or compare several models on the same molecule.
            </p>
          </Section>

          <Section id="results" title="3. Reading the Results">
            <ul className="list-disc space-y-2 pl-5" style={{ color: 'var(--text-muted)' }}>
              <li><strong>One card per endpoint.</strong> hERG and ClinTox each give a probability and the label it falls under at the stated threshold. Tox21 gives one row per assay.</li>
              <li><strong>No combined verdict.</strong> The three measurements answer different questions and are never merged into a single label, colour or number. Counting active Tox21 assays is not a severity measure.</li>
              <li><strong>Applicability.</strong> A note says when the molecule falls outside what the models were built for. Treat a result with that note as weaker.</li>
              <li><strong>Explanation.</strong> For hERG and Tox21 you can ask which atoms weighed most in the prediction. This shows what the model attended to, not a cause.</li>
              <li><strong>Unavailable endpoints.</strong> An endpoint the deployment does not serve is shown as unavailable, not left out.</li>
            </ul>
          </Section>

          <Section id="sessions" title="4. Sessions and the Agent">
            <p style={BODY_STYLE}>
              <Link to="/sessions" className="underline" style={{ color: 'var(--accent-blue)' }}>Sessions</Link> are
              where work is kept. In a session you can run analyses, ask the agent questions about a molecule, and
              follow its investigation: the trace of what it did, the evidence it read, and the report it wrote.
            </p>
            <p style={BODY_STYLE}>
              Every number in an agent answer is checked against the observation it came from before the answer is
              saved. An answer that cannot be checked is replaced by a clearly marked fallback.
            </p>
          </Section>

          <Section id="ai-connections" title="5. AI Connections">
            <p style={BODY_STYLE}>
              Prediction needs no AI account. The agent does: connect a model provider under{' '}
              <Link to="/settings" className="underline" style={{ color: 'var(--accent-blue)' }}>Settings</Link>, then
              pick the model in the session composer. Which providers can be connected depends on how the deployment
              is configured.
            </p>
          </Section>

          <Section id="limitations" title="6. Notes and Limitations of the System">
            <ul className="list-disc space-y-2 pl-5" style={{ color: 'var(--text-muted)' }}>
              <li>ToxAgent supports screening. It does not replace expert judgement or experimental data.</li>
              <li>Probabilities are model outputs and are not calibrated; read them with the threshold shown beside them.</li>
              <li>Prediction quality depends on the SMILES given and on what the models were trained on.</li>
              <li>Atom attributions are aids to interpretation, not evidence of mechanism.</li>
              <li>Literature the agent retrieves gives context and may miss relevant work.</li>
            </ul>
          </Section>
        </main>
      </div>

      <Footer />
    </div>
  );
}
