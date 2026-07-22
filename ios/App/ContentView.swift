import SwiftUI
import UniformTypeIdentifiers
import MLX
import RemixFlowKit

struct ContentView: View {
    @StateObject private var model = AppModel()
    @State private var importing = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                header
                if let name = model.songName {
                    Text(name).font(.headline)
                    steering
                    controls
                    if let status = model.status { Text(status).foregroundStyle(.secondary).font(.footnote) }
                } else {
                    ContentUnavailableView("Import a song", systemImage: "waveform",
                                           description: Text("Steer it, or let it evolve as a Living Song."))
                    Button("Import…") { importing = true }.buttonStyle(.borderedProminent)
                }
                Spacer()
            }
            .padding()
            .navigationTitle("◈ RemixFlow")
            .fileImporter(isPresented: $importing, allowedContentTypes: [.audio]) { result in
                if case .success(let url) = result { model.load(url: url) }
            }
        }
    }

    private var header: some View {
        HStack {
            Picker("", selection: $model.mode) {
                Text("Steer").tag(AppModel.Mode.steer)
                Text("∞ Living").tag(AppModel.Mode.living)
            }.pickerStyle(.segmented)
        }
    }

    private var steering: some View {
        VStack(alignment: .leading, spacing: 14) {
            slider("Variation", value: $model.steering.variationAmount, in: 0...1)
            if model.mode == .living {
                slider("Improvisation", value: $model.improvisation, in: 0...1)
            }
            TextField("Style prompt", text: $model.steering.prompt)
                .textFieldStyle(.roundedBorder)
            TextField("Lyrics (optional)", text: $model.steering.lyrics, axis: .vertical)
                .textFieldStyle(.roundedBorder).lineLimit(1...3)
        }
    }

    private var controls: some View {
        Group {
            if model.mode == .steer {
                Button(model.busy ? "Generating…" : "⚡ Generate Variation") { model.generate() }
                    .buttonStyle(.borderedProminent).disabled(model.busy)
            } else {
                Button(model.living ? "⏹ Stop Living" : "▶ Start Living") { model.toggleLiving() }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func slider(_ label: String, value: Binding<Float>, in range: ClosedRange<Float>) -> some View {
        VStack(alignment: .leading) {
            HStack { Text(label); Spacer(); Text("\(Int(value.wrappedValue * 100))%").monospacedDigit() }
                .font(.subheadline)
            Slider(value: value, in: range)
        }
    }
}

/// View-model wiring the UI to a RemixGenerator + AudioEngine.
@MainActor
final class AppModel: ObservableObject {
    enum Mode { case steer, living }
    @Published var mode: Mode = .steer
    @Published var steering = Steering()
    @Published var improvisation: Float = 0.35
    @Published var songName: String?
    @Published var status: String?
    @Published var busy = false
    @Published var living = false

    private let audio = AudioEngine()
    private var source: MLXArray?          // [2, N] @ 48 kHz
    private var generator: RemixGenerator? // set by setupModel(...)

    /// Load the model once (weights + tokenizer) from a folder laid out like the
    /// HF snapshot (transformer/ vae/ text_encoder/ condition_encoder/ tokenizer/).
    func setupModel(modelDir: URL) {
        status = "Loading model…"
        Task.detached { [weak self] in
            do {
                let store = try WeightStore(directory: modelDir)
                for c in ["transformer/", "text_encoder/", "condition_encoder/"] {
                    store.quantizeLinears(prefix: c)   // 4-bit gs32; VAE stays fp16
                }
                let tok = try await RFTokenizer(folder: modelDir.appendingPathComponent("tokenizer"))
                let pipe = ACEStepPipeline(store: store, tokenizer: tok)
                await MainActor.run { self?.generator = pipe; self?.status = "Model ready." }
            } catch {
                await MainActor.run { self?.status = "Model load failed: \(error.localizedDescription)" }
            }
        }
    }

    func load(url: URL) {
        songName = url.deletingPathExtension().lastPathComponent
        do {
            let samples = try AudioIO.load(url: url)   // [2, N] @ 48 kHz
            source = samples
            status = "Loaded \(samples.dim(1)) samples."
        } catch {
            status = "Couldn't decode audio: \(error.localizedDescription)"
        }
    }

    func generate() {
        guard let source, let generator else { status = "Model not loaded yet."; return }
        busy = true
        Task.detached { [source, steering] in
            let out = generator.generate(audio: source, steering: steering, seed: 42)
            await MainActor.run { self.audio.play(out); self.busy = false }
        }
    }

    func toggleLiving() {
        living.toggle()
        // TODO: loop — generate the next stretch and audio.enqueue() it ahead of the playhead.
    }
}
