import type { ControlSpec, ControlValue } from "../types";

interface Props {
  spec: ControlSpec;
  value: ControlValue;
  onChange: (key: string, value: ControlValue) => void;
}

/** Renders one control according to its kind (bipolar/unipolar/enum/multi). */
export function SteeringControl({ spec, value, onChange }: Props) {
  switch (spec.kind) {
    case "bipolar":
    case "unipolar":
      return <SliderControl spec={spec} value={value as number} onChange={onChange} />;
    case "enum":
      return <EnumControl spec={spec} value={value as string} onChange={onChange} />;
    case "multi":
      return <MultiControl spec={spec} value={(value as string[]) ?? []} onChange={onChange} />;
    default:
      return null;
  }
}

function SliderControl({
  spec,
  value,
  onChange,
}: {
  spec: ControlSpec;
  value: number;
  onChange: (key: string, value: ControlValue) => void;
}) {
  const bipolar = spec.kind === "bipolar";
  const min = bipolar ? -1 : 0;
  const pct = bipolar ? Math.round(((value + 1) / 2) * 100) : Math.round(value * 100);
  const isCore = spec.key === "variation_amount";
  return (
    <div className={`control slider ${isCore ? "core-control" : ""}`} title={spec.help}>
      <div className="control-head">
        <span className="control-label">{spec.label}</span>
        <span className="control-readout">
          {bipolar ? (value >= 0 ? "+" : "") : ""}
          {Math.round(value * 100)}
          {bipolar ? "" : "%"}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={1}
        step={0.01}
        value={value}
        style={{ ["--fill" as string]: `${pct}%` }}
        onChange={(e) => onChange(spec.key, Number(e.target.value))}
      />
      <div className="control-ends">
        <span>{spec.left}</span>
        <span>{spec.right}</span>
      </div>
    </div>
  );
}

function EnumControl({
  spec,
  value,
  onChange,
}: {
  spec: ControlSpec;
  value: string;
  onChange: (key: string, value: ControlValue) => void;
}) {
  return (
    <div className="control enum" title={spec.help}>
      <span className="control-label">{spec.label}</span>
      <div className="chip-row">
        {spec.options.map((opt) => (
          <button
            key={opt}
            className={`chip ${value === opt ? "active" : ""}`}
            onClick={() => onChange(spec.key, opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

function MultiControl({
  spec,
  value,
  onChange,
}: {
  spec: ControlSpec;
  value: string[];
  onChange: (key: string, value: ControlValue) => void;
}) {
  const toggle = (opt: string) => {
    const next = value.includes(opt) ? value.filter((v) => v !== opt) : [...value, opt];
    onChange(spec.key, next);
  };
  return (
    <div className="control multi" title={spec.help}>
      <span className="control-label">{spec.label}</span>
      <div className="chip-row">
        {spec.options.map((opt) => (
          <button
            key={opt}
            className={`chip ${value.includes(opt) ? "active" : ""}`}
            onClick={() => toggle(opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}
