import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL
  if (!backendUrl) {
    return NextResponse.json(
      { error: 'BACKEND_URL environment variable is not set.' },
      { status: 500 }
    )
  }

  try {
    const res = await fetch(`${backendUrl}/visit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(10_000),
    })
    if (!res.ok) {
      return NextResponse.json(
        { error: `Backend returned ${res.status}.` },
        { status: res.status }
      )
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json(
      { error: 'Failed to reach the backend.' },
      { status: 502 }
    )
  }
}
