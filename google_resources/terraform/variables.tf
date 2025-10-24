variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The GCP region (e.g., 'us-central1')."
  default     = "us-central1"
}

variable "repository_id" {
  type        = string
  description = "The name for your Artifact Registry Docker repository."
  default     = "fpl-2025"
}

variable "github_repo" {
  type        = string
  description = "Your GitHub repository in 'owner/repo_name' format (e.g., 'my-org/my-cool-app')."
  default     = "datareemz/FPL_2025"
}

variable "bucket_name" {
  type        = string
  description = "The name of the Cloud Storage bucket for FPL data."
  default     = "fpl_2025"
}

variable "secret_id" {
  type        = string
  description = "The ID of the secret to create for FPL credentials."
  default     = "fpl_2025_credentials"
}

variable "secret_data" {
  type        = string
  description = "The secret data (email and password) in JSON format: {\"email\":\"...\",\"password\":\"...\"}."
  sensitive   = true
  default     = "{\"email\":\"\",\"password\":\"\"}"
}
