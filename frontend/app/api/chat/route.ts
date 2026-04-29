import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL
  if (!backendUrl) {
    return NextResponse.json(
      { error: 'BACKEND_URL environment variable is not set.' },
      { status: 500 }
    )
  }

  const body = await req.json()

  // Check if the client wants a streaming response
  const wantsStream = body.stream === true

  if (!wantsStream) {
    // Legacy non-streaming path
    let res: Response
    try {
      res = await fetch(`${backendUrl}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60_000),
      })
    } catch (err) {
      return NextResponse.json(
        { error: 'Failed to reach the backend. Please try again.' },
        { status: 502 }
      )
    }

    if (!res.ok) {
      return NextResponse.json(
        { error: `Backend returned ${res.status}.` },
        { status: res.status }
      )
    }

    const data = await res.json()
    return NextResponse.json(data)
  }

  // Streaming path
  let res: Response
  try {
    res = await fetch(`${backendUrl}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60_000),
    })
  } catch (err) {
    return NextResponse.json(
      { error: 'Failed to reach the backend. Please try again.' },
      { status: 502 }
    )
  }

  if (!res.ok) {
    return NextResponse.json(
      { error: `Backend returned ${res.status}.` },
      { status: res.status }
    )
  }

  // Proxy the SSE stream to the client
  const reader = res.body?.getReader()
  if (!reader) {
    return NextResponse.json({ error: 'No response body.' }, { status: 502 })
  }

  const stream = new ReadableStream({
    async start(controller) {
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          controller.enqueue(value)
        }
      } catch (e) {
        controller.error(e)
      } finally {
        controller.close()
        reader.releaseLock()
      }
    },
    cancel() {
      reader.cancel()
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  })
}
