variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "github_repo" {
  type        = string
  description = "GitHub repository in format 'owner/repo-name'"
}

variable "bucket_name" {
  type        = string
  description = "The name of the Cloud Storage bucket to grant access to."
  default     = ""
}

variable "secret_name" {
  type        = string
  description = "The full resource name of the secret to grant access to."
  default     = ""
}