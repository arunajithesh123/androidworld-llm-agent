# Few-Shot Prompting Examples

## Example 1 - Adding Contact
**Goal:** Create a new contact for John Smith with phone +1234567890
- **Step 1:** Contacts app open, contact list visible → CLICK("Add contact button")
- **Step 2:** New contact form appears → TYPE("John Smith")  
- **Step 3:** Name entered, phone field focused → TYPE("+1234567890")
- **Step 4:** All required fields filled → CLICK("Save button")

## Example 2 - Sending SMS
**Goal:** Send message "Hello" to contact named Mike
- **Step 1:** Messages app open → CLICK("New message button")
- **Step 2:** New message screen → TYPE("Mike")
- **Step 3:** Contact selected → TYPE("Hello")  
- **Step 4:** Message composed → CLICK("Send button")

## Example 3 - Setting Timer
**Goal:** Set timer for 5 minutes
- **Step 1:** Clock app open → CLICK("Timer tab")
- **Step 2:** Timer interface → TYPE("5:00")
- **Step 3:** Time entered → CLICK("Start button")

## Key Patterns
- Always start with the main action button (Add, New, etc.)
- Enter data in the order it appears on screen
- Always end with Save/Send/Start to complete the action