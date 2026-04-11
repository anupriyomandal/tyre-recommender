import type { Message } from '@/app/page'

export default function MessageBubble({ message }: { message: Message }) {
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
        <p
          className="text-sm leading-relaxed text-gray-800"
          dangerouslySetInnerHTML={{ __html: message.content }}
        />
      </div>
    </div>
  )
}
