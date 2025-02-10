from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import json
import getpass

USERNAME = input("Enter your username: ")
PASSWORD = getpass.getpass("Enter your password: ")

# Initialize the webdriver (make sure to download the appropriate driver for your browser)
driver = webdriver.Chrome()

# Open the login page
driver.get("https://www.vut.cz/teacher/2/cs/zp-seznam-zadani/seznam")

# Log in
def login(driver):
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "login")))
    
    # Enter the username and password
    username_field = driver.find_element(By.NAME, "login")
    username_field.send_keys(USERNAME)
    driver.find_element(By.NAME, "btnsubmit").click()

    # Wait for the password field and enter password
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "passwd")))
    password_field = driver.find_element(By.NAME, "passwd")
    password_field.send_keys(PASSWORD)
    
    # Submit the form
    driver.find_element(By.NAME, "btnSubmit").click()
    time.sleep(10)

def navigate_to_page(driver, page_number):
    current_page = 1
    while current_page < page_number:
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "button[name='page-next']")
            if "disabled" in next_button.get_attribute("class"):
                print(f"Reached the last page at page {current_page}.")
                break
            next_button.click()
            current_page += 1
            time.sleep(5)  # Wait for the next page to load
        except Exception as e:
            print(f"Error navigating to page {current_page + 1}: {e}")
            break

def extract_and_save_data(driver):
    data = []
    
    while True:
        # Wait until the rows are loaded
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "-row-group")))

        # Find all the zadani-nazev cells (thesis links)
        links = driver.find_elements(By.CSS_SELECTOR, ".-cell.-col-zadani-nazev a")
        
        for link in links:
            href = link.get_attribute("href")
            title = link.text
            print(f"Extracting data from: {title}, {href}")
            
            # Open the link in a new tab and extract the data
            driver.execute_script("window.open(arguments[0]);", href)
            driver.switch_to.window(driver.window_handles[-1])
            
            # Wait for the page to load fully
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Extract desired data from the page
            try:
                # Find the type of thesis, supervisor, targets, and names
                name_cz = driver.find_element(By.ID, "textCZ").text
                name_en = driver.find_element(By.ID, "textEN").text
                type_of_work = driver.find_element(By.CSS_SELECTOR, ".custom-combobox-input").get_attribute("value")
                supervisor = driver.find_element(By.ID, "vedouci").get_attribute("data-defaultlabel")
                targets_cz = driver.find_element(By.ID, "cileCZ").get_attribute("value")
                targets_en = driver.find_element(By.ID, "cileEN").get_attribute("value")
                
                # Save the extracted data to the list
                data.append({
                    "Name_CZ": name_cz,
                    "Name_EN": name_en,
                    "Type of Work": type_of_work,
                    "Supervisor": supervisor,
                    "Targets_CZ": targets_cz,
                    "Targets_EN": targets_en
                })
            
            except Exception as e:
                print(f"Error extracting data for {title}: {e}")
            
            # Close the tab and switch back to the original one
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            
            driver.execute_script("window.scrollBy(0, 500);") 
            time.sleep(1)

        # Check if there is a next page button and click it
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "button[name='page-next']")
            if "disabled" in next_button.get_attribute("class"):
                break  # Break if there is no next page
            else:
                next_button.click()  # Click next page
                time.sleep(5)  # Wait for the next page to load
        except Exception as e:
            print(f"Error navigating to the next page: {e}")
            break  # Break if there was an issue finding the button

    # Write the data to a JSON file
    with open("themes.json", "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=4)

    print("JSON file created successfully!")

# Main function
if __name__ == "__main__":
    try:
        login(driver)
        navigate_to_page(driver, 0)  
        extract_and_save_data(driver)
    finally:
        driver.quit()