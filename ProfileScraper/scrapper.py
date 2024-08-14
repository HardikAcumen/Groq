from llama_index.llms.openai import OpenAI
from llama_index.core.indices.struct_store import JSONQueryEngine
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def scrape_profile(driver, profile_url):
    """Scrape required fields from LinkedIn URL"""
    driver.get("https://www.linkedin.com/uas/login")
    time.sleep(3)
    USERNAME = "hprajapati@acumen.llc"
    PASSWORD = "Hardik1234"
    email=driver.find_element(By.ID , "username")
    email.send_keys(USERNAME)
    print(email)
    password=driver.find_element( By.ID , "password")
    password.send_keys(PASSWORD)
    print(password)
    time.sleep(3)

    button = driver.find_element(By.CLASS_NAME , "btn__primary--large")
    button.click()
    print(button)
    # password.send_keys(Keys.RETURN)

    driver.get(profile_url)
 
    profile_name = driver.find_element(By.CSS_SELECTOR, "h1.text-heading-xlarge").get_attribute("innerText")
    time.sleep(10)
    profile_title = driver.find_element(By.CSS_SELECTOR, "div.text-body-medium").get_attribute("innerText")
    time.sleep(10)
    about = driver.find_element(By.CLASS_NAME , "inline-show-more-text__button inline-show-more-text__button--light link")
    time.sleep(10)
    about_text = driver.find_element(By.CLASS_NAME , "AatKoQwEMdqLKJpUYfBnnUVrTqZmSyyeIUjho").get_attribute("innerText")
    # Click on Contact Info link
    
    
    print("Profile Name: {}".format(profile_name))
    time.sleep(10)
    print("Title: {}".format(profile_title))
    time.sleep(10)

    print("Location: {}".format(about))
    time.sleep(10)


if __name__ == "__main__":
    driver = webdriver.Chrome(service=Service(r"ProfileScraper\chromedriver\chromedriver.exe") )
    
    scrape_profile(driver , profile_url=r"https://www.linkedin.com/in/mlokhandwala/")