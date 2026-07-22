import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { LivingParams, LivingSegment } from "../types";

interface Props {
  songIds: string[]; // playlist (one or more songs)
  songName: (id: string) => string;
  perSongSec: number; // how long each song plays before moving on (playlist)
  backend?: string;
  getParams: () => Omit<LivingParams, "start_index" | "start_pos">;
  onError: (msg: string) => void;
}

// Generate this many seconds ahead of the playhead (higher for playlists, since
// the first segment of a new song also pays the one-time vocal-separation cost).
const BUFFER_AHEAD = 60;

interface Scheduled {
  seg: LivingSegment;
  startTime: number;
  advance: number;
}

/** Living Mode player with gapless playback and playlist support. Segments are
 * decoded into Web Audio buffers and scheduled back-to-back on a sample-accurate
 * timeline (time-aligned crossfade tails, no gap/skip). It generates ahead of the
 * playhead; with a playlist it plays each song for ~perSongSec then transitions
 * to the next (crossfaded), looping forever when Living Repeat is on. */
export function LivingPlayer({ songIds, songName, perSongSec, backend, getParams, onError }: Props) {
  const [started, setStarted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [livingRepeat, setLivingRepeat] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [segCur, setSegCur] = useState(0);
  const [segDur, setSegDur] = useState(0);
  const [nowSong, setNowSong] = useState<string>("");

  const ctxRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const nextTimeRef = useRef(0);
  const startClockRef = useRef(0);
  const scheduledRef = useRef<Scheduled[]>([]);
  const nextStartRef = useRef(0); // window index within the current song
  const nextPosRef = useRef(0); // source position within the current song
  const pendingRef = useRef(false);
  const repeatRef = useRef(true);
  const startedRef = useRef(false);
  // Playlist generation cursor.
  const songCursorRef = useRef(0);
  const songAccumRef = useRef(0); // seconds already generated of the current song
  const songIdsRef = useRef(songIds);
  songIdsRef.current = songIds;
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
    const advance = seg.advance || buf.duration;
    const overlap = Math.max(0, buf.duration - advance);
    const startAt = first ? ctx.currentTime + 0.08 : nextTimeRef.current;
    if (first) startClockRef.current = startAt;
    const bufEnd = startAt + buf.duration;
    const advanceEnd = startAt + advance;
    if (!first && overlap > 0) {
      g.gain.setValueAtTime(0, startAt);
      g.gain.linearRampToValueAtTime(1, startAt + overlap);
    } else {
      g.gain.setValueAtTime(1, startAt);
    }
    if (overlap > 0) {
      g.gain.setValueAtTime(1, advanceEnd);
      g.gain.linearRampToValueAtTime(0, bufEnd);
    }
    src.start(startAt);
    scheduledRef.current.push({ seg, startTime: startAt, advance });
    nextTimeRef.current = advanceEnd;
  }, []);

  // Pick the song to generate next, advancing through the playlist.
  const songToGenerate = () => {
    const ids = songIdsRef.current;
    if (ids.length > 1 && songAccumRef.current >= perSongSec) {
      songCursorRef.current = (songCursorRef.current + 1) % ids.length;
      songAccumRef.current = 0;
      nextStartRef.current = 0;
      nextPosRef.current = 0;
    }
    return ids[Math.min(songCursorRef.current, ids.length - 1)];
  };

  const generateOne = useCallback(
    async (songId: string) => {
      const seg = await api.living(
        songId,
        { ...getParams(), start_index: nextStartRef.current, start_pos: nextPosRef.current },
        { backend }
      );
      nextStartRef.current = seg.next_index;
      nextPosRef.current = seg.next_pos;
      songAccumRef.current += seg.advance || seg.duration;
      await fetchAndSchedule(seg);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [backend, getParams]
  );

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
      await generateOne(songToGenerate());
      setStatus("");
    } catch (e: any) {
      onError(String(e.message ?? e));
      setStatus("");
    } finally {
      pendingRef.current = false;
      setGenerating(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generateOne, onError, perSongSec]);

  const start = useCallback(async () => {
    const ctx = ensureCtx();
    await ctx.resume();
    setStarted(true);
    startedRef.current = true;
    setPlaying(true);
    setStatus("Composing the opening…");
    nextTimeRef.current = 0;
    startClockRef.current = 0;
    scheduledRef.current = [];
    nextStartRef.current = 0;
    nextPosRef.current = 0;
    songCursorRef.current = 0;
    songAccumRef.current = 0;
    pendingRef.current = true;
    setGenerating(true);
    try {
      await generateOne(songIdsRef.current[0]);
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
  }, [generateOne, onError]);

  // Progress + buffer-fill loop.
  useEffect(() => {
    const id = window.setInterval(() => {
      const ctx = ctxRef.current;
      if (!ctx || !startedRef.current) return;
      const now = ctx.currentTime;
      const active = scheduledRef.current.filter((s) => s.startTime + s.advance > now - 1);
      scheduledRef.current = active;
      const playingSeg = active.find((s) => s.startTime <= now && now < s.startTime + s.advance);
      if (playingSeg) {
        setSegCur(now - playingSeg.startTime);
        setSegDur(playingSeg.advance);
        setNowSong(songName(playingSeg.seg.song_id));
      }
      setElapsed(Math.max(0, now - startClockRef.current));
      if (repeatRef.current && !pendingRef.current && nextTimeRef.current - now < BUFFER_AHEAD) {
        void generateNext();
      }
    }, 500);
    return () => window.clearInterval(id);
  }, [generateNext, songName]);

  useEffect(() => () => { ctxRef.current?.close().catch(() => {}); }, []);

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
  const isPlaylist = songIds.length > 1;

  return (
    <div className="living-player">
      <div className="lp-orb-wrap">
        <div className={`lp-orb ${playing ? "spinning" : ""} ${generating ? "pulsing" : ""}`} />
        <div className="lp-orb-label">
          {!started ? (isPlaylist ? `Living playlist · ${songIds.length} songs` : "Living Mode") : nowSong || "Living…"}
        </div>
      </div>

      {!started ? (
        <button className="generate lp-start" onClick={start} disabled={generating}>
          {generating ? "Composing the opening…" : isPlaylist ? "▶ Start Living Playlist" : "▶ Start Living"}
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
                <span className="tnum subtle">{isPlaylist ? nowSong : `this stretch ${fmt(segCur)} / ${fmt(segDur)}`}</span>
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
              (isPlaylist
                ? `Playlist Living — evolving each song, flowing into the next every ~${Math.round(perSongSec)}s.`
                : livingRepeat
                ? "Gapless & endless — it composes the next stretch while this one plays."
                : "Living Repeat off — stops after the buffered stretches.")}
          </p>
        </>
      )}
    </div>
  );
}
