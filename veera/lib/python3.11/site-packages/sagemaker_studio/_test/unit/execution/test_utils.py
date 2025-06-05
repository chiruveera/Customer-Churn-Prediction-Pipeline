import unittest

from sagemaker_studio.execution.utils import RemoteExecutionUtils


class TestRemoteExecutionUtils(unittest.TestCase):

    def test_pack_s3_path_for_input_file(self):
        # Test case 1: project_s3_path ends without slash, local_file_path with leading 'src/'
        project_s3_path = "s3://bucket/domain/project"
        local_file_path = "src/getting_started.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_input_file(project_s3_path, local_file_path)
        expected = "s3://bucket/domain/project/workflows/project-files/getting_started.ipynb"
        self.assertEqual(result, expected)

        # Test case 2: local_file_path without leading 'src/'
        local_file_path = "getting_started.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_input_file(project_s3_path, local_file_path)
        self.assertEqual(result, expected)

        # Test case 3: project_s3_path with a trailing slash
        project_s3_path = "s3://bucket/domain/project/"
        result = RemoteExecutionUtils.pack_s3_path_for_input_file(project_s3_path, local_file_path)
        self.assertEqual(result, expected)

        # Test case 4: local_file_path with extra sub-directory
        local_file_path = "src/folder/getting_started.ipynb"
        expected = "s3://bucket/domain/project/workflows/project-files/folder/getting_started.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_input_file(project_s3_path, local_file_path)
        self.assertEqual(result, expected)

        # Test case 5: local_file_path with the leading slash
        local_file_path = "/src/folder/getting_started.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_input_file(project_s3_path, local_file_path)
        self.assertEqual(result, expected)

    def test_pack_full_path_for_input_file(self):
        local_file_path = "src/getting_started.ipynb"
        result = RemoteExecutionUtils.pack_full_path_for_input_file(local_file_path)
        expected = "/home/sagemaker-user/src/getting_started.ipynb"
        self.assertEqual(result, expected)

        local_file_path = "/src/folder/getting_started.ipynb"
        result = RemoteExecutionUtils.pack_full_path_for_input_file(local_file_path)
        expected = "/home/sagemaker-user/src/folder/getting_started.ipynb"
        self.assertEqual(result, expected)

    def test_pack_s3_path_for_output_file(self):
        # Test case 1: s3 output location should reuse input path with leading "_" added to file name
        project_s3_path = "s3://bucket/domain/project"
        local_input_file_path = "/src/input.ipynb"
        expected_default = "s3://bucket/domain/project/workflows/output/_input.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_output_file(
            project_s3_path, local_input_file_path
        )
        self.assertEqual(result, expected_default)

        # Test case 2: input file path has a leading slash
        local_input_file_path = "/src/input.ipynb"
        expected_default = "s3://bucket/domain/project/workflows/output/_input.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_output_file(
            project_s3_path, local_input_file_path
        )
        self.assertEqual(result, expected_default)

        # Test case 3: input file path has sub-directory, should keep this sub-directory in output path
        local_input_file_path = "/src/folder/input.ipynb"
        expected_default = "s3://bucket/domain/project/workflows/output/folder/_input.ipynb"
        result = RemoteExecutionUtils.pack_s3_path_for_output_file(
            project_s3_path, local_input_file_path
        )
        self.assertEqual(result, expected_default)
