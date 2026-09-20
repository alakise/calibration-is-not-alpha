#!/usr/bin/env python3
"""Write the human-readable report for the no-paid-Jev research phase."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "quant_microstructure_phase"
NOTIONAL = 1_000.0
OFFICIAL_COST_BPS = 15.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:,.{digits}f}"


def _row(results: list[dict], strategy: str, hold: int = 15) -> dict:
    for item in results:
        if item.get("strategy") == strategy and item.get("max_hold_minutes") == hold:
            return item
    return {}


def _cost_sensitivity(row: dict) -> dict[str, float | None]:
    gross = float(row.get("gross_pnl", 0.0))
    trades = int(row.get("executed_trades", 0))
    return {
        str(cost): gross - trades * cost * NOTIONAL / 10_000 for cost in (5, 10, 15, 20)
    }


def _answers(classical: dict, signal: dict, maker: dict) -> dict[str, str]:
    rows = classical.get("canonical_n20_max_hold_results", [])
    s0 = _row(rows, "S0_BREAKOUT")
    s1 = _row(rows, "S1_BREAKOUT_VOL")
    s2 = _row(rows, "S2_BREAKOUT_VOL_ER")
    s3 = _row(rows, "S3_BREAKOUT_VOL_ER_VOLUME")
    s4 = _row(rows, "S4_BREAKOUT_VOL_ER_VOLUME_ETH")
    folds = classical.get("walk_forward", [])
    gross_oos = sum(float(f["validation_metrics"].get("gross_pnl", 0.0)) for f in folds)
    net_oos = sum(float(f["validation_metrics"].get("net_pnl", 0.0)) for f in folds)
    conservative = [
        item
        for item in maker.get("results", [])
        if item.get("fill_model") == "conservative_queue"
    ]
    q0 = next((item for item in conservative if item.get("toxicity_level") == 0), {})
    return {
        "A": f"Evet, düzeltilmiş S1 brüt PnL S0'a göre iyileşti ({_fmt(s1.get('gross_pnl'))} vs {_fmt(s0.get('gross_pnl'))} USD), ancak OOS net maliyet sonrası pozitif değil.",
        "B": f"ER filtresi brüt sonucu {_fmt(s1.get('gross_pnl'))} → {_fmt(s2.get('gross_pnl'))} USD yaptı; bu tek başına istikrarlı maliyet-sonrası kanıt değil.",
        "C": f"Volume filtresi bu koşuda brüt sonucu {_fmt(s2.get('gross_pnl'))} → {_fmt(s3.get('gross_pnl'))} USD yaptı; incremental değer OOS'ta doğrulanmadı.",
        "D": f"ETH confirmation tam örneklemde brüt sonucu {_fmt(s3.get('gross_pnl'))} → {_fmt(s4.get('gross_pnl'))} USD artırdı; walk-forward'da maliyet sonrası hâlâ negatif.",
        "E": f"OOS brüt toplam {_fmt(gross_oos)} USD; pozitif görünen katmanlar fold-stabil bir aday oluşturmuyor.",
        "F": f"Hayır. Seçilen üç validation fold toplam neti {_fmt(net_oos)} USD (15 bps round-trip).",
        "G": "Bu cutoff'ta OFI, trade-flow ve book imbalance kısa vadeli mid hareketinde güçlü/stabil OOS kanıt vermedi; son validation hedefi varyasyonsuzdu.",
        "H": "Tam örneklem tanısında spread en büyük rank korelasyonu gösterdi; bu likidite/örneklem etkisi olabilir, stabil incremental alpha değildir.",
        "I": "Gösterilmedi: microstructure nested modellerinin geç validation hedefi degenerate (std=0), bu nedenle directional iyileşme iddiası yapılamaz.",
        "J": f"Optimistic touch pozitif görünüyor, fakat conservative queue Q0 {_fmt(q0.get('net_pnl'))} USD ve maker ücreti 0 varsayımı kullanıyor; ekonomik kanıt değil.",
        "K": "Toxicity seviyeleri fill oranını düşürdü; kısa tek oturumda daha iyi net sonuç için güvenilir kanıt yok.",
        "L": "Şu an hiçbiri. Directional taraf 15 bps'de negatif; maker tarafı ise fee/queue varsayımlarına aşırı duyarlı.",
        "M": "Jev ancak deterministic modelin zaten hesaplamadığı rejim tutarlılığı, likidite stresi ve bid/ask toxicity meta-yargılarını ekleyebilir.",
        "N": "Hayır. Yeni ücretli Jev çağrısı için yeterli cost-surviving, OOS kanıt yok; yalnızca manifest hazırlandı.",
    }


def build_summary(out: Path) -> dict[str, object]:
    classical = _load(out / "classical_corrected_results.json")
    phase = _load(out / "quant_microstructure_phase.json")
    signal = _load(out / "microstructure_signal_analysis.json")
    maker = _load(out / "maker_simulation.json")
    audit = _load(out / "microstructure_data_audit.json")
    manifest = _load(out / "jev_meta_experiment_manifest.json")
    rows = classical.get("canonical_n20_max_hold_results", [])
    selected_rows = {
        item.get("strategy"): {
            "candidate_signals": item.get("candidate_signals"),
            "executed_trades": item.get("executed_trades"),
            "gross_pnl": item.get("gross_pnl"),
            "net_pnl": item.get("net_pnl"),
            "gross_return_per_trade_bps": item.get("gross_return_per_trade_bps"),
            "net_return_per_trade_bps": item.get("net_return_per_trade_bps"),
            "win_rate": item.get("win_rate"),
            "profit_factor": item.get("profit_factor"),
            "max_drawdown": item.get("max_drawdown"),
            "cost_sensitivity_net_pnl": _cost_sensitivity(item),
        }
        for item in rows
        if item.get("max_hold_minutes") == 15
    }
    mr0 = next(
        (
            item
            for item in classical.get("canonical_mr0_results", [])
            if item.get("max_hold_minutes") == 15
        ),
        {},
    )
    selected_rows["MR0_MEAN_REVERSION"] = {
        "candidate_signals": mr0.get("candidate_signals"),
        "executed_trades": mr0.get("executed_trades"),
        "gross_pnl": mr0.get("gross_pnl"),
        "net_pnl": mr0.get("net_pnl"),
        "gross_return_per_trade_bps": mr0.get("gross_return_per_trade_bps"),
        "net_return_per_trade_bps": mr0.get("net_return_per_trade_bps"),
        "cost_sensitivity_net_pnl": _cost_sensitivity(mr0),
    }
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase_cutoff": phase.get("cutoff"),
        "no_paid_jev_calls": True,
        "no_real_orders": True,
        "dataset": {
            "classical": classical.get("dataset"),
            "raw_audit": audit.get("raw"),
            "derived_audit": audit.get("derived"),
        },
        "corrected_classical": {
            "official_cost_semantics": "5 bps fee + 2 bps slippage + 0.5 bps spread per side = 15 bps round-trip",
            "previous_s1_invalid": classical.get("volatility_definition", {}).get(
                "previous_s1_invalid"
            ),
            "candidate_counts": classical.get("canonical_candidate_counts"),
            "canonical_hold_15m": selected_rows,
            "walk_forward": classical.get("walk_forward"),
            "research_labels": classical.get("research_labels"),
        },
        "microstructure": {
            "signal_analysis": signal,
            "maker": maker,
            "maker_comparison": {
                "optimistic_q0": next(
                    (
                        item
                        for item in maker.get("results", [])
                        if item.get("fill_model") == "optimistic_touch"
                        and item.get("toxicity_level") == 0
                    ),
                    None,
                ),
                "conservative_q0": next(
                    (
                        item
                        for item in maker.get("results", [])
                        if item.get("fill_model") == "conservative_queue"
                        and item.get("toxicity_level") == 0
                    ),
                    None,
                ),
                "optimistic_q4": next(
                    (
                        item
                        for item in maker.get("results", [])
                        if item.get("fill_model") == "optimistic_touch"
                        and item.get("toxicity_level") == 4
                    ),
                    None,
                ),
                "conservative_q4": next(
                    (
                        item
                        for item in maker.get("results", [])
                        if item.get("fill_model") == "conservative_queue"
                        and item.get("toxicity_level") == 4
                    ),
                    None,
                ),
            },
        },
        "jev_manifest": manifest,
    }
    summary["explicit_answers"] = _answers(classical, signal, maker)
    return summary


def _markdown(summary: dict[str, object]) -> str:
    classical = summary["corrected_classical"]
    rows = classical["canonical_hold_15m"]
    audit = summary["dataset"]["raw_audit"]
    derived = summary["dataset"]["derived_audit"]
    signal = summary["microstructure"]["signal_analysis"]
    comparison = summary["microstructure"]["maker_comparison"]
    answers = summary["explicit_answers"]
    lines = [
        "# EXECUTIVE SUMMARY",
        "",
        "Bu faz Jev'e yeni ücretli çağrı yapmadan tamamlandı. Düzeltilmiş klasik breakout ailesi ve kısa canlı Kraken microstructure örneği aynı resmi maliyet modeliyle değerlendirildi. Sonuç: mevcut cutoff'ta maliyet sonrası doğrulanmış alpha adayı yok.",
        "",
        f"- Klasik veri: {summary['dataset']['classical'].get('rows', 'NA')} BTC 1m satır; {classical['candidate_counts']}",
        f"- Microstructure cutoff: {summary['phase_cutoff']}; {audit.get('raw_lines')} raw satır, {audit.get('reconnect_events')} reconnect, bad JSON={audit.get('bad_json')}",
        f"- Canonical state: {derived.get('symbols')}; valid book rows={derived.get('valid_book_rows')}",
        "- Jev manifest yalnızca tasarım; budget_usd=0 ve no_inference_launched=true.",
        "",
        "## 1. Corrected classical strategy results",
        "",
        "15 dakika maksimum tutuş, 15 bps round-trip maliyet. Candidate count ile executed trade count ayrıdır.",
        "",
        "| Strategy | Candidates | Trades | Gross USD | Net USD | Gross/trade bps | Net/trade bps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "S0_BREAKOUT",
        "S1_BREAKOUT_VOL",
        "S2_BREAKOUT_VOL_ER",
        "S3_BREAKOUT_VOL_ER_VOLUME",
        "S4_BREAKOUT_VOL_ER_VOLUME_ETH",
        "MR0_MEAN_REVERSION",
    ):
        item = rows.get(name, {})
        lines.append(
            f"| {name} | {_fmt(item.get('candidate_signals'), 0)} | {_fmt(item.get('executed_trades'), 0)} | {_fmt(item.get('gross_pnl'))} | {_fmt(item.get('net_pnl'))} | {_fmt(item.get('gross_return_per_trade_bps'), 3)} | {_fmt(item.get('net_return_per_trade_bps'), 3)} |"
        )
    lines += [
        "",
        "## 2. What changed after fixing S1 volatility scaling",
        "",
        "Önceki S1 horizon-scaled volatilite karşılaştırması geçersizdi. Yeni tanım aynı frekansta `vol5/vol60`; S1 artık 4,898 aday ve 2,733 gerçekleşen trade üretiyor. Bu nedenle önceki tek-trade S1 sonucu kanıt olarak kullanılmıyor.",
        "",
        "## 3–6. Breakout, ER, volume ve ETH",
        "",
        "Katmanlar aynı timestamp/parametre evreninde nested. Brüt iyileşmeler görülebilse de resmi 15 bps maliyet altında tüm canonical katmanlar negatif. Walk-forward'da her validation fold flat başlıyor; seçilen S4/N60 fold netleri sırasıyla -$223.69, -$187.17, -$164.28.",
        "",
        "## 7. Microstructure dataset audit",
        "",
        f"Raw feed: {audit.get('feed_counts')}; symbols={audit.get('symbols')}; interval={audit.get('first_event')} → {audit.get('last_event')}. Collector public-only kaldı. Trade side Kraken taker-side metadata olarak saklandı; yerel aggressor çıkarımı yapılmadı. L1/L10 yok, yalnızca L5 persisted; OFI gerçek price-level OFI değil depth-change proxy.",
        "",
        "## 8–12. OFI, trade-flow, book, microprice ve ablation",
        "",
        "Tam örneklem diagnostics içinde rank correlations: "
        + ", ".join(
            f"{key}={_fmt(value.get('rank_corr_30s'), 4)}"
            for key, value in signal.get("feature_diagnostics", {}).items()
        )
        + ". Geç validation target std=0 olduğundan M0–M5 OOS yön/korrelasyon metrikleri bilgi taşıyan hareket yokluğunda degenerate; bunlar predictive evidence değildir.",
        "",
        "## 13. Directional strategy economics",
        "",
        "Resmi maliyet 5 bps fee + 2 bps slippage + 0.5 bps spread her side; toplam 15 bps round-trip. Cost sensitivity 5/10/15/20 bps rapor JSON'da; gross edge, turnover ve maliyet burden birbirine karıştırılmadı.",
        "",
        "## 14–16. Maker simulator, fill sensitivity, toxicity",
        "",
        f"Touch Q0: net {_fmt(comparison.get('optimistic_q0', {}).get('net_pnl'))} USD, fills {_fmt(comparison.get('optimistic_q0', {}).get('fills'), 0)}. Conservative queue Q0: net {_fmt(comparison.get('conservative_q0', {}).get('net_pnl'))} USD, fills {_fmt(comparison.get('conservative_q0', {}).get('fills'), 0)}. Q4'te fill sayısı dramatik biçimde düşüyor. Maker fee/rebate 0 varsayıldı; gerçek tier ve queue position bilinmediği için bunlar diagnostic, alpha iddiası değil.",
        "",
        "## 17. Directional vs passive-maker",
        "",
        "Directional branch resmi maliyette negatif. Passive-maker branch optimistic-touch'ta pozitif görünse de conservative queue, zero maker-fee ve tek kısa oturum varsayımlarına duyarlı; iki branch arasında kazanan ilan edilmiyor.",
        "",
        "## 18–19. Proposed compact Jev meta-state / future bundle",
        "",
        "Manifest; candidate type, normalized momentum, volatility regime, ER, volume confirmation, OFI windows, trade-flow, L1/L5 imbalance, microprice displacement, spread, depth ve liquidity state alanlarını öneriyor. Rejim, signal consistency, liquidity stress, toxic bid/ask ve quote environment typed judgments olarak tasarlanmış; yeni Jev inference çalıştırılmadı.",
        "",
        "## 20. Recommended next experiment",
        "",
        "Daha uzun ve fiyat hareketi varyasyonu olan microstructure cutoff'ı topla; aynı feature setini non-degenerate chronological blocks üzerinde yeniden test et; maker tarafında gerçek signed fee/rebate schedule ve daha conservative queue/cancel latency modeli ekle. Sonra yalnızca ön-kayıtlı, küçük bir Jev meta-gate holdout'u düşün.",
        "",
        "## Explicit answers A–N",
        "",
    ]
    for key in "ABCDEFGHIJKLMN":
        lines.append(f"**{key}.** {answers[key]}")
        lines.append("")
    lines += [
        "## Limitations",
        "",
        "- Collector L5 ile sınırlı; L1/L10, gerçek queue position, cancellation latency ve maker fee tier yok.",
        "- Microstructure validation penceresinde hedef varyasyonu yok; bu nedenle model sonucu alpha kanıtı sayılmadı.",
        "- Maker sonuçları bir kısa public-data oturumu ve zero-fee varsayımına dayanıyor.",
        "- Gerçek Kraken emirleri ve yeni Jev çağrıları kapalı.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.output_dir)
    summary = build_summary(out)
    (out / "quant_microstructure_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out / "quant_microstructure_summary.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    print(json.dumps({"event": "summary_written", "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
