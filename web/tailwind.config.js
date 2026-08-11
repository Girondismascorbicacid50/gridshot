/** 2046 Print Shop design tokens mapped into Tailwind. Single source of truth. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    colors: {
      transparent: "transparent",
      // substrates
      paper: "#E9E3D3",
      "paper-2": "#DED7C2",
      field: "#17191C",
      navy: "#243049",
      cobalt: "#1F3C74",
      knockout: "#F3EFE4",
      // spot inks
      orange: "#C8531E",
      red: "#C13322",
      teal: "#246E72",
      gold: "#EFA92E",
      blue: "#2F6FB0",
      // extended
      olive: "#4B5337",
      yellow: "#F2C21A",
      water: "#2E6E8E",
      // neutrals
      line: "#B9B19C",
      muted: "#625D50",
      "orange-text": "#9B3C17",
    },
    fontFamily: {
      display: ['"Space Grotesk"', "sans-serif"],
      body: ['"IBM Plex Sans"', "sans-serif"],
      mono: ['"IBM Plex Mono"', "monospace"],
    },
    fontSize: {
      xs: "0.75rem",
      sm: "0.875rem",
      base: "1rem",
      lg: "1.25rem",
      xl: "1.563rem",
      "2xl": "1.953rem",
      "3xl": "2.441rem",
      "4xl": "3.052rem",
    },
    borderRadius: { none: "0", DEFAULT: "2px", full: "9999px" },
    extend: {
      spacing: { 18: "4.5rem", 22: "5.5rem" },
      maxWidth: { container: "1120px" },
    },
  },
  plugins: [],
};
