import AVFoundation
import Foundation
import MLX

/// Minimal audio playback + gapless Living scheduling via AVAudioEngine.
/// Feed it stereo Float samples [2, N] at 48 kHz.
final class AudioEngine: ObservableObject {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sampleRate: Double = 48_000
    @Published var isPlaying = false

    init() {
        engine.attach(player)
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        engine.connect(player, to: engine.mainMixerNode, format: fmt)
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        try? AVAudioSession.sharedInstance().setActive(true)
        try? engine.start()
    }

    private func buffer(from samples: MLXArray) -> AVAudioPCMBuffer {
        // samples: [2, N] float. MLX → interleaved-by-channel PCM buffer.
        let n = AVAudioFrameCount(samples.dim(1))
        let fmt = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n)!
        buf.frameLength = n
        let host = samples.asType(.float32)
        for ch in 0..<2 {
            let row = host[ch].asArray(Float.self)
            row.withUnsafeBufferPointer { src in
                buf.floatChannelData![ch].update(from: src.baseAddress!, count: Int(n))
            }
        }
        return buf
    }

    /// Play a one-shot clip.
    func play(_ samples: MLXArray) {
        player.stop()
        player.scheduleBuffer(buffer(from: samples), at: nil)
        player.play()
        isPlaying = true
    }

    /// Living Mode: enqueue the next stretch to play gaplessly after the current.
    /// The scheduler starts each buffer exactly when the previous ends.
    func enqueue(_ samples: MLXArray) {
        player.scheduleBuffer(buffer(from: samples), at: nil)
        if !player.isPlaying { player.play(); isPlaying = true }
    }

    func stop() { player.stop(); isPlaying = false }
}
