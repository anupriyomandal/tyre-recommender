'use client'

import { useState, useRef, useEffect } from 'react'
import MessageBubble from '@/components/MessageBubble'
import ChatInput from '@/components/ChatInput'

export type Message = {
  role: 'user' | 'assistant'
  content: string
}

const WELCOME_MESSAGE: Message = {
  role: 'assistant',
  content:
    "Hi! I'm your <b>CEAT Tyre Advisor</b>. Tell me your vehicle's make and model and I'll recommend the right tyres for you.",
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE])
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (query: string) => {
    const userMsg: Message = { role: 'user', content: query }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setLoading(true)

    // Build history excluding the static welcome message
    const history = updatedMessages
      .slice(1) // skip the welcome
      .slice(0, -1) // skip the message we just added (it's the current query)
      .map((m) => ({ role: m.role, content: m.content }))

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history }),
      })

      const data = await res.json()
      const answer = data.answer ?? data.error ?? 'Something went wrong. Please try again.'

      setMessages((prev) => [...prev, { role: 'assistant', content: answer }])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Could not reach the server. Please try again.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([WELCOME_MESSAGE])
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-ceat-red flex items-center justify-center">
            <TyreIcon />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-gray-900">CEAT Tyre Advisor</h1>
            <p className="text-xs text-gray-500">Find the perfect tyre for your vehicle</p>
          </div>
        </div>
        <button
          onClick={clearChat}
          className="text-xs text-gray-400 hover:text-gray-600 transition-colors px-2 py-1 rounded hover:bg-gray-100"
        >
          Clear chat
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto chat-scroll px-4 py-4 space-y-4">
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 bg-white px-4 py-3">
        <ChatInput onSend={sendMessage} disabled={loading} />
        <p className="text-center text-xs text-gray-400 mt-2">
          Ask about any vehicle — e.g. "Honda City", "Toyota Fortuner", "Hyundai Creta"
        </p>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex items-end gap-2 animate-fade-in">
      <div className="w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0">
        <TyreIcon small />
      </div>
      <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
        <div className="flex gap-1 items-center h-4">
          <span className="typing-dot w-2 h-2 rounded-full bg-gray-400 block" />
          <span className="typing-dot w-2 h-2 rounded-full bg-gray-400 block" />
          <span className="typing-dot w-2 h-2 rounded-full bg-gray-400 block" />
        </div>
      </div>
    </div>
  )
}

function TyreIcon({ small }: { small?: boolean }) {
  const size = small ? 14 : 18
  return (
    <svg
      width={size}
      height={size}
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
  )
}
