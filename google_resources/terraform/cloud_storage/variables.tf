variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The GCP region (e.g., 'us-central1')."
}

variable "bucket_name" {
  type        = string
  description = "The name of the Cloud Storage bucket."
}

variable "storage_class" {
  type        = string
  description = "The storage class for the bucket (e.g., STANDARD, NEARLINE, COLDLINE)."
  default     = "STANDARD"
}

variable "versioning_enabled" {
  type        = bool
  description = "Enable versioning for the bucket."
  default     = true
}

variable "lifecycle_age_days" {
  type        = number
  description = "Number of days after which objects are deleted."
  default     = 30
}