import type { Message } from '@/app/page'

// Known CEAT tyre models — sorted longest first so multi-word names match first
const CEAT_MODELS = [
  'SportDrive SUV',
  'SportDrive',
  'SecuraDrive SUV',
  'SecuraDrive SD004',
  'SecuraDrive',
  'Milaze X5',
  'Milaze X3',
  'Milaze LT',
  'Milaze',
  'CrossDrive AT',
  'CrossDrive HT',
  'CrossDrive',
  'EnergyDrive EV',
  'EnergyDrive',
  'SD004',
].sort((a, b) => b.length - a.length)

const MODEL_PATTERN = CEAT_MODELS.map((m) => m.replace(/\s+/g, '\\s+')).join('|')
const TYRE_PATTERN = new RegExp(
  `(\\d{3}/\\d{2,3}R\\d{2})\\s+(${MODEL_PATTERN})(?:\\s+(TL|TT))?`,
  'gi'
)

function colorizeTyres(html: string): string {
  return html.replace(
    TYRE_PATTERN,
    (_match, size: string, model: string, suffix: string | undefined) => {
      const suffixHtml = suffix ? ` ${suffix}` : ''
      return `<span class="text-ceat-orange">${size}</span> <span class="text-ceat-blue">${model}</span>${suffixHtml}`
    }
  )
}

export default function MessageBubble({
  message,
  isStreaming = false,
}: {
  message: Message
  isStreaming?: boolean
}) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[80%] bg-ceat-blue text-white rounded-2xl rounded-br-sm px-4 py-2.5 shadow-sm">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    )
  }

  const formattedContent = colorizeTyres(message.content)

  return (
    <div className="flex items-end gap-2 animate-fade-in">
      <div className="w-7 h-7 rounded-full bg-ceat-blue flex items-center justify-center flex-shrink-0">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="12" r="3" />
          <line x1="12" y1="2" x2="12" y2="9" />
          <line x1="12" y1="15" x2="12" y2="22" />
          <line x1="2" y1="12" x2="9" y2="12" />
          <line x1="15" y1="12" x2="22" y2="12" />
        </svg>
      </div>
      <div className="max-w-[80%] bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-2.5 shadow-sm">
        {isStreaming && !message.content ? (
          <div className="flex gap-1 items-center h-4">
            <span className="typing-dot w-2 h-2 rounded-full bg-gray-400 block" />
            <span className="typing-dot w-2 h-2 rounded-full bg-gray-400 block" />
            <span className="typing-dot w-2 h-2 rounded-full bg-gray-400 block" />
          </div>
        ) : (
          <p
            className="text-sm leading-relaxed text-gray-800"
            dangerouslySetInnerHTML={{ __html: formattedContent }}
          />
        )}
      </div>
    </div>
  )
}
