---
license: cc-by-4.0

extra_gated_prompt: "In order to access and utilize the MMFakeBench dataset, you are required to consent to the terms outlined below:

(1) The MMFakeBench dataset is for non-commercial research purposes only.
(2) You and your affiliated institution must agree not to reproduce, duplicate, copy, sell, trade, resell or exploit any portion of the images or any derived data from the dataset for any purpose.
(3) You and your affiliated institution must agree not to further copy, publish, or distribute any portion of the MMFakeBench dataset or any derived data from the dataset for any purpose.
(4) You and your affiliated institution take full responsibilities of any consequence as a result of using the MMFakeBench dataset, and shall defend and indemnify the authors or the authors’ affiliated institutions against any and all claims arising from such uses.
(5) The use of MMFakeBench dataset in publications must cite the citation given below.
(6) The authors reserve the right to terminate your access to the MMFakeBench dataset at any time.

By agreeing you accept to share your contact information (email and username) with the repository authors.
"
extra_gated_fields:
  First Name: text
  Last Name: text
  Affiliation/Organization: text
  Email: text
  Principal Investigator/Advisor's Name: text
  Country: country
  Specific date: date_picker
  I want to use this dataset for:
    type: select
    options:
      - Research
      - Education
      - label: Other
        value: other
  I agree to use this dataset for non-commercial use ONLY: checkbox
configs:
  - config_name: MMFakeBench_val
    data_files: MMFakeBench_val.json
  - config_name: MMFakeBench_test
    data_files: MMFakeBench_test.json

---
