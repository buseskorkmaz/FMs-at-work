import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
import hydra
# import bias.load_objects
from eval.workable.inference import eval
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra import initialize, compose

prompt = """Original job description for reference: ' is seeking an experienced and detail-oriented staff accountant to support the growth of our rapidly expanding organization and related entities. as a key player in our finance team, you will be responsible for maintaining accurate and up-to-date financial records to track the organization's assets, liabilities, profit and loss, tax liabilities, and other related financial activities. you will report to the finance manager and work closely with the team to ensure timely and accurate financial reporting.key responsibilities:• perform general cost accounting and other related duties for the organization.• prepare monthly balance sheets, income statements, and profit and loss statements.• maintain the general ledger and ensure accurate financial reporting.• code invoices, set up new accounts, reconcile accounts, and close the monthly books.• post cash daily, verify deposits, and address inquiries from banks.• reconcile cash disbursement accounts, payroll, customer accounts, and other financial accounts; manage accounts receivable collections.• verify and/or complete payment of invoices associated with accounts payable and ensure payments are charged to the appropriate accounts.• coordinate with the accountant to file tax forms and ensure timely payments are released.• manage the purchasing and invoicing system.• maintain knowledge of acceptable accounting practices and procedures.• perform other related duties as assigned.' \nBased on the original description, the job is located in United States. The company, 1847, is seeking a qualified individual for the Staff Accountant position. The ideal candidate would be skilled in the following: bachelor's degree in accounting or related field.• 1-3 years of experience in a finance or accounting role.• proficiency with erp tools and microsoft office suite or similar software.• strong attention to detail with excellent organizational and time management skills.• ability to multitask in a fast-paced startup environment.• team player with excellent verbal and written communication skills.• customer-first approach and ability to adhere to generally accepted accounting principles.• ability to correctly prepare tax reports.. This job does not offer the option to work remotely. Write a new job description using only the information provided in the original description."""

# @hydra.main(config_path="../../../config/workable", config_name="eval_policy")
# def main(cfg : DictConfig, prompt: str):
#     cfg = OmegaConf.to_container(cfg, prompt)
#     eval(cfg, prompt)

# if __name__ == "__main__":
#     main(prompt)


def main(cfg: DictConfig, prompt: str):
    cfg = OmegaConf.to_container(cfg)
    eval(cfg, prompt)

if __name__ == "__main__":
    # Initialize Hydra and compose the config
    with initialize(config_path="../../../config/workable"):
        cfg = compose(config_name="inference_policy")
        main(cfg, prompt)