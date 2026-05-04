'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import MessageBubble from '@/components/MessageBubble'
import ChatInput from '@/components/ChatInput'
import CeatLogo from '@/components/CeatLogo'

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
  const [partialAnswer, setPartialAnswer] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, partialAnswer])

  // Record a page visit once on load
  useEffect(() => {
    fetch('/api/visit', { method: 'POST' }).catch(() => {})
  }, [])

  const sendMessage = useCallback(async (query: string) => {
    const userMsg: Message = { role: 'user', content: query }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)
    setLoading(true)
    setPartialAnswer('')

    // Build history excluding the static welcome message
    const history = updatedMessages
      .slice(1) // skip the welcome
      .slice(0, -1) // skip the message we just added (it's the current query)
      .map((m) => ({ role: m.role, content: m.content }))

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history, stream: true }),
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.error || 'Something went wrong. Please try again.' },
        ])
        return
      }

      const reader = res.body?.getReader()
      if (!reader) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Could not read response. Please try again.' },
        ])
        return
      }

      const decoder = new TextDecoder()
      let fullAnswer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') {
              setLoading(false)
              setPartialAnswer('')
              setMessages((prev) => [...prev, { role: 'assistant', content: fullAnswer }])
              return
            }
            if (data.startsWith('Error:')) {
              setLoading(false)
              setPartialAnswer('')
              setMessages((prev) => [...prev, { role: 'assistant', content: data }])
              return
            }
            // Unescape newlines
            const text = data.replace(/\\n/g, '\n')
            fullAnswer += text
            setPartialAnswer(fullAnswer)
          }
        }
      }

      // If stream ended without [DONE], commit whatever we have
      setLoading(false)
      setPartialAnswer('')
      if (fullAnswer) {
        setMessages((prev) => [...prev, { role: 'assistant', content: fullAnswer }])
      }
    } catch {
      setLoading(false)
      setPartialAnswer('')
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Could not reach the server. Please try again.' },
      ])
    }
  }, [messages])

  const clearChat = () => {
    setMessages([WELCOME_MESSAGE])
    setPartialAnswer('')
    setLoading(false)
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200 shadow-sm">
        <div className="flex items-center gap-3">
          <CeatLogo className="h-8 w-auto" />
          <p className="text-xs text-gray-500">Tyre Advisor</p>
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
        {(loading || partialAnswer) && (
          <MessageBubble
            key="streaming"
            message={{ role: 'assistant', content: partialAnswer || '' }}
            isStreaming={loading && !partialAnswer}
          />
        )}
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
