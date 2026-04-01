Language Playground is an interactive application in which users can analyze text to determine:

- The primary language the text is written in.
- Any personally identifiable information, such as names, phone numbers, email addresses, and physical addresses.

# Visual layout and style

The app is an interactive playground with two panes:

- A pane on the left in which the user can type or upload and view a text file for analysis. At the top right of this pane, there is a paperclip "attach" button that the user can click to upload a .txt file (maximum size 10k), and a "Detect" button that they can click to start the analysis.
- A pane on the left in which the results of the analysis are shown. The format of the results depends on the specific kind of analysis:
  - For language detection, the results simply report the detected primary language (for example "French" or "Simplified Chinese")
  - For PII detection, the results should show a list of the terms that were redacted, and what kind of information they represent (name, phone number, email address, or physical address). After PII detection, the original text in the left pane shou;d be redisplayed with the detected terms redacted (replaced by the placeholders [PERSON], [PHONENUMBER], [EMAIL], and [ADDRESS]).

Above the pane on the left, there is a left-aligned "Language Playground" heading, under which there is a drop-down list in which the user can choose between "Language detection" and "Text PII extraction".

By default, the app should open with the "Language detection" option selected and the text "Add text or a file to detect language" below a black and white text file icon in the center of the right pane. wSwitching to the "Text PII extraction" option resets the UI and the text in the center of the right pane should be "Add text or a file to detect PII".

The color scheme should be similar to that of the chat-playground app, with purple highlights and the same light/dark mode themes that exist in that app.

# Language detection functionality

Users can type or upload plain text in the left pane.

When the user clicks "Detect", the app should detect the primary language of the sample text from the following list of languages:

- English
- French
- Spanish
- Portuguese
- German
- Italian
- Simplified Chinese
- Japanese
- Hindi
- Arabic
- Russian

To determine the language, the app should maintain an internal text-based dataset of the most common 100 everyday words in each of those languages;and use a combination of the statistical frequency of these terms in the text, and its character encoding.

When the language has been detected, the source text should become read-only and the app should display the results in the right pane indicating both the language name (for example, "French") and the 2-character ISO language code (for example, "fr"). At the top of the right pane (above the language name and code) there should be a circular gauge indicating the confidence in the prediction (based on the presence of any terms from other languages found in the text)

After the text has been analyzed and the results displayed, the attach button should be disabled and the "Detect" button should change to say "Edit". Clicking it re-enables the attach button and makes the source text editable so the user can change it and re-detect. If the user attaches a new file, the current source text should all be replaced and the results cleared.

# PII redaction functionality

Users can type or upload plain text in the left pane.

When the user clicks "Detect", the app should use a combination of Compromise.js and regular expression matching to identify:

- Names of people
- Phone numbers
- Email addresses
- Street addresses (from door number to postal code or country)

After detecting the PII terms, the app should redisplay the source text with the PII terms redacted and replaced by the placeholders [PERSON], [PHONENUMBER], [EMAIL], and [ADDRESS]. In the right pane, the redacted terms should be listed along with their information type - for example:

- John Smith (Person)
- 555 123 4567 (Phone)
- <john.smith@contoso.com> (Email)
- 123 Anystreet, Anytown, WA, USA, 01234 (Address)

After the text has been analyzed and the results displayed, the attach button should be disabled and the "Detect" button should change to say "Edit". Clicking it re-enables the attach button and makes the source text editable so the user can change it and re-detect. If the user attaches a new file, the current source text should all be replaced and the results cleared.

# Switching analyzers

When the user selects "Language detection" or "Text PII extraction" in the drop-down list above the left pane, the UI should reset; removing any text and results, and if necessary, re-enabling the attach button and switching the "Edit" button back to "Generate".

# Accessibility

All UI elements should have appropriate ARIA settings and tooltips.
The UI must be keyboard-navigable with a logical tab-sequence.

# Security

The app should escape HTML in text strings to mitigate cross-site scripting attacks.
