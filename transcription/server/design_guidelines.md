# Design Guidelines: Audio Transcription Web Application

## Design Approach
**System-Based Approach**: Material Design principles for clean, functional utility application
- Focus on clarity, efficiency, and user feedback
- Emphasize workflow states and progressive disclosure
- Professional, trustworthy aesthetic appropriate for productivity tools

## Core Design Principles
1. **Workflow Clarity**: Each step (upload → processing → results) visually distinct
2. **Immediate Feedback**: Real-time state changes and progress indicators
3. **Result Emphasis**: Transcription output as primary focus once generated

## Typography
**Font Family**: Inter or Roboto via Google Fonts CDN
- **Headings**: 600 weight, 1.5rem - 2rem
- **Body Text**: 400 weight, 1rem (results display)
- **Labels**: 500 weight, 0.875rem
- **Metadata**: 400 weight, 0.875rem (file info, timestamps)

## Layout System
**Spacing Units**: Tailwind units of 3, 4, 6, 8, 12
- Container: max-w-4xl centered
- Section padding: py-8 to py-12
- Component spacing: gap-6 between major sections
- Internal padding: p-4 to p-6 for cards/containers

## Component Structure

### Upload Area
- Large, centered dropzone (min-h-64)
- Dashed border indicating drop area
- Upload icon (cloud-up or upload from Heroicons)
- Primary CTA button "Select Audio File"
- Supported formats indicator (MP3, WAV, M4A, etc.)
- File size limit notice

### File Preview Card (After Upload)
- Compact horizontal layout
- Audio icon + filename + duration + file size
- Remove/Replace button
- Subtle background to differentiate from upload state

### Processing State
- Linear progress bar or spinner
- "Transcribing..." status text
- Estimated time remaining (if available)
- Semi-transparent overlay over file preview

### Results Display
- Full-width text container with max-w-prose for readability
- White/neutral background card with subtle shadow
- Monospace font option toggle for technical accuracy
- Copy-to-clipboard button (top-right of results card)
- Download as TXT option
- Word count and processing time metadata
- Clear visual hierarchy: results are the hero element

### Action Buttons
- Primary: "Transcribe" or "Upload Another File"
- Secondary: "Copy Text", "Download"
- Icon + text labels for clarity
- Full-width on mobile, inline on desktop

## Visual Hierarchy States

**State 1 - Empty/Upload Ready**
- Upload area dominates (60% viewport height)
- Centered, inviting appearance
- Clear instructions and file format support

**State 2 - File Selected**
- Upload area collapses to compact preview
- "Start Transcription" button becomes prominent
- File details clearly displayed

**State 3 - Processing**
- Progress indicator front and center
- File preview visible but de-emphasized
- Cancel option available

**State 4 - Results**
- Results container becomes primary focus
- Takes 70% of vertical space
- File info moves to header/metadata area
- Clear "New Transcription" action to restart

## Accessibility
- ARIA labels for upload dropzone
- Keyboard navigation for all interactive elements
- Focus indicators on all clickable areas
- High contrast text in results display
- Screen reader announcements for state changes

## Icons
Use Heroicons via CDN:
- CloudArrowUp: Upload state
- DocumentText: File preview
- ClipboardDocument: Copy action
- ArrowDownTray: Download action
- XMark: Remove file
- CheckCircle: Success state

## Animations
**Minimal, Purposeful Only**:
- Smooth height transitions when states change (transition-all duration-300)
- Progress bar animation during processing
- Subtle fade-in for results (fade-in-up)
- No decorative animations

## Images
**No hero images required** - This is a utility application focused on functionality, not visual storytelling. The interface itself is the focus.