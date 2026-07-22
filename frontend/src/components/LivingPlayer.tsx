import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { LivingParams, LivingSegment } from "../types";

interface Props {
  songId: string;
  backend?: string;
  getParams: () => Omit<LivingParams, "start_index" | "start_pos">;
  onError: (msg: string) => void;
}

// Keep this many seconds of audio scheduled ahead of the playhead.
const BUFFER_AHEAD = 45;

interface Scheduled {
  seg: LivingSegment;
  startTime: number; // AudioContext time this segment begins
  advance: number; // timeline advance before the next segment begins
}

/** Living Mode player with gapless playback. Segments are decoded into Web Audio
 * buffers and scheduled back-to-back on a sample-accurate timeline, so there's
 * no pause between them. New segments are generated ahead of the playhead and
 * appended; "Living Repeat" keeps the buffer filled so it never ends. */
export function LivingPlayer({ songId, backend, getParams, onError }: Props) {
  const [started, setStarted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [livingRepeat, setLivingRepeat] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [segCur, setSegCur] = useState(0);
  const [segDur, setSegDur] = useState(0);

  const ctxRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const nextTimeRef = useRef(0); // ctx time where the current buffer ends
  const startClockRef = useRef(0); // ctx time the performance began
  const scheduledRef = useRef<Scheduled[]>([]);
  const nextStartRef = useRef(0); // window index (tension/seed phase) to continue from
  const nextPosRef = useRef(0); // source position (sec) to continue from
  const pendingRef = useRef(false);
  const repeatRef = useRef(true);
  const startedRef = useRef(false);
  repeatRef.current = livingRepeat;

  const ensureCtx = () => {
    if (!ctxRef.current) {
      const ctx = new AudioContext();
      const gain = ctx.createGain();
      gain.connect(ctx.destination);
      ctxRef.current = ctx;
      gainRef.current = gain;
    }
    return ctxRef.current!;
  };

  const scheduleBuffer = useCallback((buf: AudioBuffer, seg: LivingSegment) => {
    const ctx = ensureCtx();
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    src.connect(g);
    g.connect(gainRef.current!);
    const first = startClockRef.current === 0;
    // The timeline advances by `advance`; the buffer is `advance + crossfade`
    // long. The tail overlaps the NEXT segment's head over the SAME source
    // moment (both rendered it), so this is a time-aligned crossfade — no skip.
    const advance = seg.advance || buf.duration;
    const overlap = Math.max(0, buf.duration - advance);
    // startAt = where this segment begins = previous segment's advance-end.
    const startAt = first ? ctx.currentTime + 0.08 : nextTimeRef.current;
    if (first) startClockRef.current = startAt;
    const bufEnd = startAt + buf.duration;
    const advanceEnd = startAt + advance;
    // Fade in over the head (aligns with the previous segment's tail fade-out).
    if (!first && overlap > 0) {
      g.gain.setValueAtTime(0, startAt);
      g.gain.linearRampToValueAtTime(1, startAt + overlap);
    } else {
      g.gain.setValueAtTime(1, startAt);
    }
    // Fade out over the tail (overlaps the next segment's head).
    if (overlap > 0) {
      g.gain.setValueAtTime(1, advanceEnd);
      g.gain.linearRampToValueAtTime(0, bufEnd);
    }
    src.start(startAt);
    scheduledRef.current.push({ seg, startTime: startAt, advance });
    nextTimeRef.current = advanceEnd; // next segment starts exactly here
  }, []);

  const fetchAndSchedule = useCallback(
    async (seg: LivingSegment) => {
      const ctx = ensureCtx();
      const arr = await fetch(api.livingAudioUrl(seg.audio_url)).then((r) => r.arrayBuffer());
      const buf = await ctx.decodeAudioData(arr);
      scheduleBuffer(buf, seg);
    },
    [scheduleBuffer]
  );

  const generateNext = useCallback(async () => {
    if (pendingRef.current) return;
    pendingRef.current = true;
    setGenerating(true);
    setStatus("Composing the next stretch…");
    try {
      const seg = await api.living(
        songId,
        { ...getParams(), start_index: nextStartRef.current, start_pos: nextPosRef.current },
        { backend }
      );
      nextStartRef.current = seg.next_index;
      nextPosRef.current = seg.next_pos;
      await fetchAndSchedule(seg);
      setStatus("");
    } catch (e: any) {
      onError(String(e.message ?? e));
      setStatus("");
    } finally {
      pendingRef.current = false;
      setGenerating(false);
    }
  }, [songId, backend, getParams, fetchAndSchedule, onError]);

  const start = useCallback(async () => {
    const ctx = ensureCtx();
    await ctx.resume();
    setStarted(true);
    startedRef.current = true;
    setPlaying(true);
    setStatus("Composing the opening…");
    // Reset performance state.
    nextTimeRef.current = 0;
    startClockRef.current = 0;
    scheduledRef.current = [];
    nextStartRef.current = 0;
    nextPosRef.current = 0;
    pendingRef.current = true;
    setGenerating(true);
    try {
      const seg = await api.living(songId, { ...getParams(), start_index: 0, start_pos: 0 }, { backend });
      nextStartRef.current = seg.next_index;
      nextPosRef.current = seg.next_pos;
      await fetchAndSchedule(seg);
      setStatus("");
    } catch (e: any) {
      onError(String(e.message ?? e));
      setStarted(false);
      startedRef.current = false;
      setPlaying(false);
      setStatus("");
    } finally {
      pendingRef.current = false;
      setGenerating(false);
    }
  }, [songId, backend, getParams, fetchAndSchedule, onError]);

  // Progress + buffer-fill loop.
  useEffect(() => {
    const id = window.setInterval(() => {
      const ctx = ctxRef.current;
      if (!ctx || !startedRef.current) return;
      const now = ctx.currentTime;
      // Prune finished segments; find the one playing now (by its advance window).
      const active = scheduledRef.current.filter((s) => s.startTime + s.advance > now - 1);
      scheduledRef.current = active;
      const playingSeg = active.find((s) => s.startTime <= now && now < s.startTime + s.advance);
      if (playingSeg) {
        setSegCur(now - playingSeg.startTime);
        setSegDur(playingSeg.advance);
      }
      setElapsed(Math.max(0, now - startClockRef.current));
      // Keep the buffer filled ahead of the playhead.
      const bufferedAhead = nextTimeRef.current - now;
      if (repeatRef.current && !pendingRef.current && bufferedAhead < BUFFER_AHEAD) {
        void generateNext();
      }
    }, 500);
    return () => window.clearInterval(id);
  }, [generateNext]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      ctxRef.current?.close().catch(() => {});
    };
  }, []);

  const togglePlay = async () => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (playing) {
      await ctx.suspend();
      setPlaying(false);
    } else {
      await ctx.resume();
      setPlaying(true);
    }
  };

  const fmt = (s: number) =>
    isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "0:00";

  return (
    <div className="living-player">
      <div className="lp-orb-wrap">
        <div className={`lp-orb ${playing ? "spinning" : ""} ${generating ? "pulsing" : ""}`} />
        <div className="lp-orb-label">
          {!started ? "Living Mode" : playing ? "Living…" : "Paused"}
        </div>
      </div>

      {!started ? (
        <button className="generate lp-start" onClick={start} disabled={generating}>
          {generating ? "Composing the opening…" : "▶ Start Living"}
        </button>
      ) : (
        <>
          <div className="lp-transport">
            <button className="play lp-play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
              {playing ? (
                <svg viewBox="0 0 24 24" fill="currentColor" width="22" height="22"><path d="M6 5h4v14H6zM14 5h4v14h-4z" /></svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="currentColor" width="22" height="22"><path d="M8 5v14l11-7z" /></svg>
              )}
            </button>
            <div className="lp-seek-wrap">
              <div className="lp-seek">
                <div className="lp-seek-fill" style={{ width: `${segDur ? (segCur / segDur) * 100 : 0}%` }} />
              </div>
              <div className="lp-time">
                <span className="tnum">total {fmt(elapsed)}</span>
                <span className="tnum subtle">this stretch {fmt(segCur)} / {fmt(segDur)}</span>
              </div>
            </div>
            <button
              className={`lp-repeat ${livingRepeat ? "on" : ""}`}
              onClick={() => setLivingRepeat((v) => !v)}
              title="Living Repeat — never-ending, never-repeating"
            >
              ∞
            </button>
          </div>
          <p className="subtle lp-status">
            {status ||
              (livingRepeat
                ? "Gapless & endless — it composes the next stretch while this one plays."
                : "Living Repeat off — stops after the buffered stretches.")}
          </p>
        </>
      )}
    </div>
  );
}
