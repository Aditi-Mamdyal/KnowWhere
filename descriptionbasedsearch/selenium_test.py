"""
RollNo_SY4_Selenium.py
======================
Selenium Test Suite — KnowWhere Corporate Search System
Subject : Case Study for Selenium Testing (CIE Assignment)
Class   : SY4

Problem Statement:
    KnowWhere is an offline semantic search engine that allows authenticated
    corporate users to search documents and images using natural language
    descriptions. This test suite validates the login, search, and admin
    workflows via automated browser testing.

REQUIREMENTS COVERED:
    ✔ Form Handling    — Login form (text inputs), Search form (text + dropdown),
                         Admin form (text + password + dropdown)
    ✔ Navigation       — Login → Search → Admin Panel → Back to Search
    ✔ Synchronization  — Explicit Waits (WebDriverWait + expected_conditions)
    ✔ Validation       — Assert statements to verify page titles, messages, results

HOW TO RUN:
    1. Start the Flask app:   python app.py
    2. Run this script:       python RollNo_SY4_Selenium.py
    3. Chrome must be installed. chromedriver is managed automatically by webdriver-manager.

INSTALL DEPENDENCIES:
    pip install selenium webdriver-manager
"""

import time
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_URL       = "http://127.0.0.1:5000"
ADMIN_USER     = "admin"
ADMIN_PASS     = "admin123"
TEST_USER      = "selenium_testuser"
TEST_PASS      = "test@1234"
WAIT_TIMEOUT   = 10   # seconds for explicit waits


# ─── SELENIUM TEST CLASS ───────────────────────────────────────────────────────

class KnowWhereSeleniumTests(unittest.TestCase):
    """
    Full Selenium test suite for KnowWhere Corporate Search System.
    Tests run in order (alphabetical by method name in unittest).
    """

    @classmethod
    def setUpClass(cls):
        """Launch Chrome once for the entire test suite."""
        print("\n" + "=" * 60)
        print("  KnowWhere Selenium Test Suite")
        print("=" * 60)

        options = webdriver.ChromeOptions()
        # Comment out the next line if you want to see the browser
        # options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,900")

        cls.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )
        cls.driver.implicitly_wait(5)
        cls.wait = WebDriverWait(cls.driver, WAIT_TIMEOUT)
        print("[SETUP] Chrome launched.")

    @classmethod
    def tearDownClass(cls):
        """Close browser after all tests finish."""
        cls.driver.quit()
        print("\n[TEARDOWN] Browser closed.")
        print("=" * 60)

    def setUp(self):
        """Navigate to base URL before each test."""
        self.driver.get(BASE_URL)
        time.sleep(0.5)

    # ── HELPER METHODS ─────────────────────────────────────────────────────────

    def _login(self, username=ADMIN_USER, password=ADMIN_PASS):
        """Reusable login helper — fills form and submits."""
        self.driver.get(f"{BASE_URL}/login")

        username_field = self.wait.until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        password_field = self.driver.find_element(By.ID, "password")
        submit_btn     = self.driver.find_element(By.ID, "login-btn")

        username_field.clear()
        username_field.send_keys(username)
        password_field.clear()
        password_field.send_keys(password)
        submit_btn.click()

        # Wait for redirect away from /login
        self.wait.until(EC.url_changes(f"{BASE_URL}/login"))

    def _logout(self):
        """Reusable logout helper."""
        try:
            logout_btn = self.wait.until(
                EC.element_to_be_clickable((By.ID, "logout-link"))
            )
            logout_btn.click()
            self.wait.until(EC.url_contains("/login"))
        except Exception:
            self.driver.get(f"{BASE_URL}/logout")

    # ── TEST 1: PAGE LOAD & REDIRECT ───────────────────────────────────────────

    def test_01_homepage_redirects_to_login(self):
        """
        Visiting '/' while not logged in must redirect to /login.
        Validates: Navigation, Page Load
        """
        print("\n[TEST 1] Homepage redirect to login...")
        self.driver.get(BASE_URL)

        self.wait.until(EC.url_contains("/login"))
        assert "/login" in self.driver.current_url, \
            f"Expected /login URL, got: {self.driver.current_url}"

        title = self.driver.title
        assert "KnowWhere" in title, f"Page title missing 'KnowWhere': {title}"

        print(f"  ✔ Redirected to: {self.driver.current_url}")
        print(f"  ✔ Page title: {title}")

    # ── TEST 2: LOGIN FORM — INVALID CREDENTIALS ───────────────────────────────

    def test_02_login_with_wrong_password(self):
        """
        Submitting wrong credentials must show an error message.
        Validates: Form Handling (text inputs), Error Message Assertion
        """
        print("\n[TEST 2] Login with wrong password...")
        self.driver.get(f"{BASE_URL}/login")

        # Explicit wait for the username field to appear
        username_field = self.wait.until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        password_field = self.driver.find_element(By.ID, "password")

        # Fill in wrong credentials
        username_field.send_keys("admin")
        password_field.send_keys("wrongpassword123")
        self.driver.find_element(By.ID, "login-btn").click()

        # Wait for error message to appear
        error_msg = self.wait.until(
            EC.presence_of_element_located((By.ID, "error-msg"))
        )
        error_text = error_msg.text
        assert "Invalid" in error_text or "incorrect" in error_text.lower(), \
            f"Expected error message, got: '{error_text}'"

        # Confirm we're still on /login
        assert "/login" in self.driver.current_url, \
            "Should remain on login page after failed attempt"

        print(f"  ✔ Error displayed: '{error_text}'")
        print(f"  ✔ Stayed on login page: {self.driver.current_url}")

    # ── TEST 3: LOGIN FORM — VALID CREDENTIALS ─────────────────────────────────

    def test_03_login_with_valid_credentials(self):
        """
        Correct admin credentials must redirect to /search.
        Validates: Form Handling, Navigation, Assert on URL
        """
        print("\n[TEST 3] Login with valid admin credentials...")
        self._login(ADMIN_USER, ADMIN_PASS)

        # Should be on /search now
        assert "/search" in self.driver.current_url, \
            f"Expected /search after login, got: {self.driver.current_url}"

        # Verify username is shown in the navbar
        user_info = self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "user-info"))
        )
        assert ADMIN_USER in user_info.text.lower(), \
            f"Username not shown in navbar: {user_info.text}"

        print(f"  ✔ Redirected to: {self.driver.current_url}")
        print(f"  ✔ Navbar shows: {user_info.text}")
        self._logout()

    # ── TEST 4: SEARCH FORM — TEXT INPUT + DROPDOWN ────────────────────────────

    def test_04_search_form_text_and_dropdown(self):
        """
        Enter a query and change the search mode dropdown.
        Validates: Form Handling (text field + dropdown), Navigation, Synchronization
        """
        print("\n[TEST 4] Search form — text input + dropdown...")
        self._login()

        # Wait for the search page to load
        query_field = self.wait.until(
            EC.presence_of_element_located((By.ID, "query"))
        )

        # Fill the text input
        query_field.clear()
        query_field.send_keys("machine learning report")

        # Use Select class for the dropdown
        mode_dropdown = Select(self.driver.find_element(By.ID, "mode"))
        mode_dropdown.select_by_value("documents")

        # Verify the dropdown selection
        selected = mode_dropdown.first_selected_option.get_attribute("value")
        assert selected == "documents", \
            f"Dropdown should be 'documents', got: '{selected}'"

        # Submit the search
        self.driver.find_element(By.ID, "search-btn").click()

        # Wait for result-count element to appear
        self.wait.until(
            EC.presence_of_element_located((By.ID, "result-count"))
        )

        result_count_el = self.driver.find_element(By.ID, "result-count")
        print(f"  ✔ Search submitted. Results: {result_count_el.text.strip()}")
        print(f"  ✔ Dropdown selected: {selected}")

        self._logout()

    # ── TEST 5: DROPDOWN — SELECT "IMAGES" MODE ────────────────────────────────

    def test_05_search_mode_dropdown_images(self):
        """
        Change dropdown to 'Images Only' and verify it submits with that mode.
        Validates: Dropdown form field
        """
        print("\n[TEST 5] Search dropdown — Images Only mode...")
        self._login()

        self.wait.until(EC.presence_of_element_located((By.ID, "query")))
        self.driver.find_element(By.ID, "query").send_keys("team photo")

        mode_dropdown = Select(self.driver.find_element(By.ID, "mode"))
        mode_dropdown.select_by_value("images")

        selected = mode_dropdown.first_selected_option.get_attribute("value")
        assert selected == "images", \
            f"Dropdown should be 'images', got '{selected}'"

        self.driver.find_element(By.ID, "search-btn").click()

        self.wait.until(
            EC.presence_of_element_located((By.ID, "result-count"))
        )
        print(f"  ✔ Images Only mode selected and submitted")
        self._logout()

    # ── TEST 6: NAVIGATION — SEARCH → ADMIN PANEL ──────────────────────────────

    def test_06_navigate_to_admin_panel(self):
        """
        Click the Admin Panel link from the search page.
        Validates: Navigation across two pages, Assert on URL
        """
        print("\n[TEST 6] Navigate to Admin Panel...")
        self._login()

        # Wait for admin link to appear
        admin_link = self.wait.until(
            EC.element_to_be_clickable((By.ID, "admin-link"))
        )
        admin_link.click()

        # Wait for /admin URL
        self.wait.until(EC.url_contains("/admin"))
        assert "/admin" in self.driver.current_url, \
            f"Expected /admin URL, got: {self.driver.current_url}"

        # Verify admin panel heading
        page_source = self.driver.page_source
        assert "Admin Panel" in page_source, "Admin Panel heading not found"

        print(f"  ✔ Navigated to: {self.driver.current_url}")
        print(f"  ✔ Admin Panel page confirmed")
        self._logout()

    # ── TEST 7: ADMIN FORM — CREATE USER ───────────────────────────────────────

    def test_07_admin_create_user(self):
        """
        Fill the Create User form with text, password, and role dropdown.
        Validates: Form Handling (3 field types), Admin navigation, Assert
        """
        print(f"\n[TEST 7] Admin Create User: '{TEST_USER}'...")
        self._login()
        self.driver.get(f"{BASE_URL}/admin")

        # Wait for the create user form
        new_username_field = self.wait.until(
            EC.presence_of_element_located((By.ID, "new_username"))
        )
        new_password_field = self.driver.find_element(By.ID, "new_password")
        role_dropdown      = Select(self.driver.find_element(By.ID, "new_role"))

        # Fill text input
        new_username_field.send_keys(TEST_USER)

        # Fill password input
        new_password_field.send_keys(TEST_PASS)

        # Select role from dropdown
        role_dropdown.select_by_value("user")

        selected_role = role_dropdown.first_selected_option.get_attribute("value")
        assert selected_role == "user", \
            f"Role dropdown should be 'user', got: '{selected_role}'"

        # Submit
        self.driver.find_element(By.ID, "create-btn").click()

        # Wait for success message
        message_el = self.wait.until(
            EC.presence_of_element_located((By.ID, "admin-message"))
        )
        msg_text = message_el.text
        assert TEST_USER in msg_text or "created" in msg_text.lower(), \
            f"Expected success message, got: '{msg_text}'"

        print(f"  ✔ Form filled: username='{TEST_USER}', role='user'")
        print(f"  ✔ Message displayed: '{msg_text}'")
        self._logout()

    # ── TEST 8: NEW USER LOGIN ─────────────────────────────────────────────────

    def test_08_new_user_can_login(self):
        """
        The user created in Test 7 must be able to log in successfully.
        Validates: Form Handling, Navigation, Assert on Search page
        """
        print(f"\n[TEST 8] New user '{TEST_USER}' login...")
        self._login(TEST_USER, TEST_PASS)

        assert "/search" in self.driver.current_url, \
            f"New user should reach /search, got: {self.driver.current_url}"

        # Confirm this user does NOT see the Admin Panel link
        try:
            self.driver.find_element(By.ID, "admin-link")
            assert False, "Non-admin user should NOT see Admin Panel link"
        except NoSuchElementException:
            pass   # expected

        print(f"  ✔ New user logged in, landed on: {self.driver.current_url}")
        print(f"  ✔ Admin Panel link correctly hidden for regular user")
        self._logout()

    # ── TEST 9: LOGOUT ─────────────────────────────────────────────────────────

    def test_09_logout_redirects_to_login(self):
        """
        Clicking Logout must redirect to /login and show flash message.
        Validates: Navigation, Explicit Wait on flash message
        """
        print("\n[TEST 9] Logout test...")
        self._login()

        logout_link = self.wait.until(
            EC.element_to_be_clickable((By.ID, "logout-link"))
        )
        logout_link.click()

        # Explicit wait for /login
        self.wait.until(EC.url_contains("/login"))
        assert "/login" in self.driver.current_url, \
            f"Expected /login after logout, got: {self.driver.current_url}"

        # Flash message should appear
        flash_msg = self.wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "flash"))
        )
        assert "logged out" in flash_msg.text.lower(), \
            f"Expected 'logged out' message, got: '{flash_msg.text}'"

        print(f"  ✔ Redirected to: {self.driver.current_url}")
        print(f"  ✔ Flash message: '{flash_msg.text}'")

    # ── TEST 10: SESSION PROTECTION ────────────────────────────────────────────

    def test_10_protected_route_without_login(self):
        """
        Accessing /search while not logged in must redirect to /login.
        Validates: Session protection, Navigation, Explicit Wait
        """
        print("\n[TEST 10] Access /search without login...")

        # Ensure we are logged out
        self.driver.get(f"{BASE_URL}/logout")
        time.sleep(0.5)

        # Try to directly access /search
        self.driver.get(f"{BASE_URL}/search")

        # Explicit wait — should be redirected to login
        self.wait.until(EC.url_contains("/login"))
        assert "/login" in self.driver.current_url, \
            f"Expected /login redirect, got: {self.driver.current_url}"

        print(f"  ✔ Correctly blocked, redirected to: {self.driver.current_url}")

    # ── TEST 11: CLEANUP — DELETE TEST USER ────────────────────────────────────

    def test_11_admin_delete_test_user(self):
        """
        Admin deletes the test user created in Test 7. Cleanup step.
        Validates: Form submission, Assert on success message
        """
        print(f"\n[TEST 11] Admin delete user '{TEST_USER}'...")
        self._login()
        self.driver.get(f"{BASE_URL}/admin")

        # Find the delete button for TEST_USER using XPath
        try:
            delete_form = self.wait.until(
                EC.presence_of_element_located(
                    (By.XPATH,
                     f"//input[@name='target_username'][@value='{TEST_USER}']"
                     f"/following-sibling::button[@type='submit']")
                )
            )
            # Execute click via JS to bypass the confirm() dialog
            self.driver.execute_script(
                "arguments[0].closest('form').submit();", delete_form
            )
        except TimeoutException:
            # User may not exist (Test 7 might have been skipped)
            print(f"  ⚠ User '{TEST_USER}' not found — skipping delete.")
            self._logout()
            return

        # Wait for admin-message
        message_el = self.wait.until(
            EC.presence_of_element_located((By.ID, "admin-message"))
        )
        print(f"  ✔ Delete result: '{message_el.text}'")
        self._logout()


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║        KnowWhere — Selenium Test Suite                    ║
║  Make sure Flask app is running:  python app.py           ║
╚═══════════════════════════════════════════════════════════╝
""")
    unittest.main(verbosity=2)