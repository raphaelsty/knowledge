/* Money formatting — every UI surface that shows balance / cost /
 * price MUST go through here so the display stays consistent if we
 * ever change the rules.
 *
 * Internal accounting unit: integer cents. The Postgres column is
 * still named `credits` for historical reasons, with the invariant
 * `1 credit == 1 cent`; the UI never surfaces the word "credits"
 * and always renders in USD. EU users can still pay in EUR via
 * Polar's hosted checkout — Polar handles the FX on their side and
 * we receive the USD-equivalent amount.
 *
 * Exposed on `window.KnowledgeMoney` so plain-script consumers
 * (profile/credits/, profile/storage/, export/, search/page.js)
 * can pick it up without an import.
 */
(function () {
  "use strict";

  /** Format a cents amount as a USD string, e.g. 850 → "$8.50". */
  function fmt(cents) {
    if (cents == null || Number.isNaN(Number(cents))) return "$0";
    const n = Number(cents);
    const sign = n < 0 ? "-" : "";
    const abs = Math.abs(n) / 100;
    // Trim trailing zeros for whole-dollar amounts ("$5" not "$5.00")
    // but keep two decimals for fractional cents ("$1.03").
    const formatted = abs % 1 === 0 ? `$${abs}` : `$${abs.toFixed(2)}`;
    return `${sign}${formatted}`;
  }

  /** Format an integer USD-dollar amount, e.g. 5 → "$5". */
  function fmtDollars(dollars) {
    return fmt(Math.round(Number(dollars || 0) * 100));
  }

  /** Difference between two cent amounts, formatted with sign, e.g.
   *  "+$0.20" for a bonus. Returns empty string for zero. */
  function fmtBonus(cents) {
    if (!cents) return "";
    const sign = cents > 0 ? "+" : "−";
    return `${sign}${fmt(Math.abs(cents))}`;
  }

  window.KnowledgeMoney = { fmt, fmtDollars, fmtBonus };
})();
