/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'fpl-purple': '#37003c',
                'fpl-green': '#00ff87',
                'fpl-pink': '#e90052',
                'fpl-cyan': '#00ff85',
            },
        },
    },
    plugins: [],
}
