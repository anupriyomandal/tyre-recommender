'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled: boolean
}

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [value])

  const handleSubmit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex items-end gap-2 bg-gray-100 rounded-2xl px-3 py-2">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Ask about your vehicle..."
        rows={1}
        className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-800 placeholder-gray-400 leading-relaxed disabled:opacity-50"
        style={{ maxHeight: '160px' }}
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
        className="flex-shrink-0 w-8 h-8 rounded-xl bg-ceat-blue flex items-center justify-center transition-opacity disabled:opacity-30 hover:opacity-90"
        aria-label="Send"
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="white"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <line x1="22" y1="2" x2="11" y2="13" />
          <polygon points="22 2 15 22 11 13 2 9 22 2" />
        </svg>
      </button>
    </div>
  )
}
