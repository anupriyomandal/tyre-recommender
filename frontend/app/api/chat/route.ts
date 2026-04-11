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
