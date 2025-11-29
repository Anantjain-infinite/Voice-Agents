import { Button } from '@/components/livekit/button';

function WelcomeImage() {
  return (
    <svg
      width="64"
      height="64"
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="text-fg0 mb-4 size-16"
    >
      {/* D20 Dice Icon */}
      <path
        d="M32 4L8 16L4 32L8 48L32 60L56 48L60 32L56 16L32 4Z"
        fill="currentColor"
        opacity="0.2"
      />
      <path
        d="M32 4L8 16L32 28L56 16L32 4Z"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
      />
      <path
        d="M32 28V60M8 16V48M56 16V48M4 32H8M56 32H60M8 48L32 60L56 48"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="32" cy="32" r="6" fill="currentColor" />
    </svg>
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center">
        <WelcomeImage />

        <h1 className="text-foreground mb-2 text-2xl font-bold">
          Voice Game Master
        </h1>

        <p className="text-foreground max-w-prose pt-1 leading-6 font-medium">
          Embark on an epic fantasy adventure in the realm of Eldergrove
        </p>

        <p className="text-muted-foreground max-w-md pt-2 text-sm leading-5">
          Your Game Master awaits to guide you through a tale of mystery, danger, and discovery.
          Speak your actions and shape the story with your voice.
        </p>

        <Button variant="primary" size="lg" onClick={onStartCall} className="mt-6 w-64 font-mono">
          🎲 {startButtonText}
        </Button>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center">
        <p className="text-muted-foreground max-w-prose pt-1 text-xs leading-5 font-normal text-pretty md:text-sm">
          Built with{' '}
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://docs.livekit.io/agents/start/voice-ai/"
            className="underline"
          >
            LiveKit Voice AI
          </a>
          {' '}• Your adventure begins with a single word
        </p>
      </div>
    </div>
  );
};