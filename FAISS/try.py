# import logging.config
# import logging
# logging.config.fileConfig('try.conf')
# logger = logging.getLogger('simpleExample')


# i = 5

# if (i > 5):
#     logger.info(f"It is {i>5} tha i greater than 5")
# else:
#     logger.info(f"It is {i>5} tha i greater than 5")

import logging
import logging.config



def perform_operation(value):
    if value < 0:
        raise ValueError("Invalid value: Value cannot be negative.")
    else:
        # Continue with normal execution
        logging.info("Operation performed successfully.")


try:
    input_value = int(input("Enter a value: "))
    perform_operation(input_value)
except ValueError as ve:
    logging.exception("Exception occurred: %s", str(ve))
