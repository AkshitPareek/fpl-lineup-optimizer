// Minimal stroke icon set (24px grid, 1.75 stroke) so we don't pull in an icon library.

function Icon({ size = 18, children, className, ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

export const IconRoute = p => (
  <Icon {...p}><circle cx="6" cy="19" r="2.5" /><circle cx="18" cy="5" r="2.5" /><path d="M8.5 19H16a3.5 3.5 0 0 0 0-7H8a3.5 3.5 0 0 1 0-7h7.5" /></Icon>
)
export const IconBolt = p => (
  <Icon {...p}><path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" /></Icon>
)
export const IconStar = p => (
  <Icon {...p}><path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9L12 3Z" /></Icon>
)
export const IconHistory = p => (
  <Icon {...p}><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" /></Icon>
)
export const IconChart = p => (
  <Icon {...p}><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></Icon>
)
export const IconCrown = p => (
  <Icon {...p}><path d="m3 7 4.5 4L12 5l4.5 6L21 7l-2 11H5L3 7Z" /></Icon>
)
export const IconChip = p => (
  <Icon {...p}><circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4.5" /><path d="M12 3.5v3M12 17.5v3M3.5 12h3M17.5 12h3" /></Icon>
)
export const IconSwap = p => (
  <Icon {...p}><path d="M7 4 3 8l4 4" /><path d="M3 8h13" /><path d="m17 20 4-4-4-4" /><path d="M21 16H8" /></Icon>
)
export const IconShield = p => (
  <Icon {...p}><path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.2 7.5 9.5 4.3-1.3 7.5-4.9 7.5-9.5V6L12 3Z" /></Icon>
)
export const IconUser = p => (
  <Icon {...p}><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></Icon>
)
export const IconPitch = p => (
  <Icon {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 12h18" /><circle cx="12" cy="12" r="2.5" /></Icon>
)
export const IconList = p => (
  <Icon {...p}><path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01" /></Icon>
)
export const IconX = p => (
  <Icon {...p}><path d="M18 6 6 18M6 6l12 12" /></Icon>
)
export const IconTrendUp = p => (
  <Icon {...p}><path d="m3 17 6-6 4 4 8-8" /><path d="M15 7h6v6" /></Icon>
)
export const IconTrendDown = p => (
  <Icon {...p}><path d="m3 7 6 6 4-4 8 8" /><path d="M15 17h6v-6" /></Icon>
)
export const IconSparkle = p => (
  <Icon {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" /></Icon>
)
export const IconTarget = p => (
  <Icon {...p}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></Icon>
)
export const IconPound = p => (
  <Icon {...p}><path d="M17 19H7c1.5-1 2.5-3 2.5-5.5V9a3.5 3.5 0 0 1 6.5-1.8" /><path d="M6.5 13H14" /></Icon>
)
export const IconPlay = p => (
  <Icon {...p}><path d="M7 4.5v15l12-7.5-12-7.5Z" /></Icon>
)
