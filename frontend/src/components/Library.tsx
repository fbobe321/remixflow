import { useRef, useState } from "react";
import type { Song } from "../types";

interface Props {
  songs: Song[];
  currentSongId: string | null;
  busy: boolean;
  onImport: (file: File, title: string) => void;
  onSelect: (songId: string) => void;
  onDelete: (songId: string) => void;
}

const ACCEPT = ".mp3,.wav,.flac,.ogg";

/** Song import (PRD §1) + library of imported tracks. */
export function Library({ songs, currentSongId, busy, onImport, onSelect, onDelete }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [drag, setDrag] = useState(false);

  const pick = (file: File | undefined) => {
    if (!file) return;
    const title = file.name.replace(/\.[^.]+$/, "");
    onImport(file, title);
  };

  return (
    <div className="library">
      <div
        className={`dropzone ${drag ? "drag" : ""} ${busy ? "busy" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <div className="dz-icon">🎵</div>
        <div>{busy ? "Analyzing…" : "Drop a song or click to import"}</div>
        <div className="subtle">MP3 · WAV · FLAC · OGG</div>
      </div>

      <div className="song-list">
        {songs.length === 0 && <p className="subtle">No songs yet.</p>}
        {songs.map((s) => (
          <div
            key={s.id}
            className={`song-row ${s.id === currentSongId ? "active" : ""}`}
            onClick={() => onSelect(s.id)}
          >
            <span className="song-title">{s.title}</span>
            <button
              className="song-del"
              title="Delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.id);
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
